"""Generate Codex subagents (``.codex/agents/*.toml``) from ``.claude/agents/*.md``.

Usage::

    uv run python scripts/sync_agents.py [--check]   # --check: exit 1 when stale

The Markdown body becomes ``developer_instructions`` verbatim. The sandbox is read-only
unless the frontmatter says ``codex-sandbox: workspace-write`` or the tools include an
editing tool (then the session sandbox is inherited).

@domain tooling
"""

import argparse
import json
import sys
from pathlib import Path
from typing import ClassVar

import yaml

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

SOURCE_DIR = Path(".claude") / "agents"
"""Claude Code subagents, the single source (relative to the root)."""

TARGET_DIR = Path(".codex") / "agents"
"""Generated Codex subagents (relative to the root)."""

SKIPPED = {"VENDORED.md"}
"""Files in the source directory that are not agents."""

WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
"""Tools that let an agent change files; without them the agent is read-only."""

SANDBOX_KEY = "codex-sandbox"
"""Frontmatter key that overrides the generated Codex ``sandbox_mode``."""

SANDBOX_MODES = {"read-only", "workspace-write"}
"""Values ``codex-sandbox`` may take."""

HEADER = (
    "# Generated from {source} by scripts/sync_agents.py; do not edit.\n"
    "# Change the Markdown file and run: uv run python scripts/sync_agents.py\n"
)
"""First lines of every generated file."""


class AgentFileError(ValueError):
    """An agent Markdown file cannot be converted."""

    REASONS: ClassVar[dict[str, str]] = {
        "no-frontmatter": "has no frontmatter",
        "not-mapping": "frontmatter is not a mapping",
        "not-closed": "frontmatter is not closed",
        "bad-name": "name must equal the file name",
        "no-description": "description is missing",
        "bad-sandbox": "codex-sandbox must be read-only or workspace-write",
    }
    """Message per reason code."""

    def __init__(self, source: str, reason: str) -> None:
        """Build the message from the file name and a reason code."""
        super().__init__(f"{source}: {self.REASONS[reason]}")


def split_agent(text: str, source: str = "agent") -> tuple[dict[str, object], str]:
    """Split an agent Markdown file into its frontmatter mapping and body.

    Args:
        text: Content of ``.claude/agents/<name>.md``.
        source: File name used in error messages.

    Returns:
        The parsed frontmatter and the body with surrounding blank lines removed.

    Raises:
        AgentFileError: The file has no frontmatter or it is not a mapping.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise AgentFileError(source, "no-frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            meta = yaml.safe_load("\n".join(lines[1:index])) or {}
            if not isinstance(meta, dict):
                raise AgentFileError(source, "not-mapping")
            body = "\n".join(lines[index + 1 :]).strip("\n")
            return meta, body + "\n"
    raise AgentFileError(source, "not-closed")


def tools_of(meta: dict[str, object]) -> set[str]:
    """Return the agent's tool names from the comma-separated ``tools`` field."""
    tools = (tool.strip() for tool in str(meta.get("tools", "")).split(","))
    return {tool for tool in tools if tool}


def sandbox_of(meta: dict[str, object], source: str) -> str | None:
    """Return the Codex ``sandbox_mode`` for an agent, or ``None`` to inherit it.

    Raises:
        AgentFileError: ``codex-sandbox`` has a value outside ``SANDBOX_MODES``.
    """
    if SANDBOX_KEY in meta:
        value = meta[SANDBOX_KEY]
        if value not in SANDBOX_MODES:
            raise AgentFileError(source, "bad-sandbox")
        return str(value)
    return None if tools_of(meta) & WRITE_TOOLS else "read-only"


def toml_string(value: str) -> str:
    """Return ``value`` as a single-line TOML basic string."""
    return json.dumps(value, ensure_ascii=False)


def toml_multiline(value: str) -> str:
    """Return ``value`` as a TOML multi-line string that round-trips exactly.

    A literal string (``'''``) keeps backslashes as written; it is used unless the
    text itself contains ``'''``, in which case a basic string with escapes is used.
    The opening delimiter is followed by a newline, which TOML trims.
    """
    if "'''" not in value:
        return "'''\n" + value + "'''"
    escaped = value.replace("\\", "\\\\").replace('"""', '""\\"')
    return '"""\n' + escaped + '"""'


def render(source: Path, root: Path) -> tuple[str, str]:
    """Return the agent name and the Codex TOML for one agent file.

    Args:
        source: Path of the Markdown agent file.
        root: Repository root, used for the header's relative path.

    Returns:
        ``(name, toml_text)``.

    Raises:
        AgentFileError: ``name`` or ``description`` is missing, or ``name`` differs from
            the file name.
    """
    meta, body = split_agent(source.read_text(encoding="utf-8"), source.name)
    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or name != source.stem:
        raise AgentFileError(source.name, "bad-name")
    if not isinstance(description, str) or not description.strip():
        raise AgentFileError(source.name, "no-description")
    lines = [
        HEADER.format(source=source.relative_to(root).as_posix()),
        f"name = {toml_string(name)}",
        f"description = {toml_string(description.strip())}",
    ]
    sandbox = sandbox_of(meta, source.name)
    if sandbox is not None:
        lines.append(f"sandbox_mode = {toml_string(sandbox)}")
    lines.append(f"developer_instructions = {toml_multiline(body)}")
    return name, "\n".join(lines) + "\n"


def build(root: Path) -> dict[Path, str]:
    """Return every generated Codex agent path with its content."""
    outputs: dict[Path, str] = {}
    for source in sorted((root / SOURCE_DIR).glob("*.md")):
        if source.name in SKIPPED:
            continue
        name, text = render(source, root)
        outputs[root / TARGET_DIR / f"{name}.toml"] = text
    return outputs


def generated_files(root: Path) -> list[Path]:
    """Return existing TOML files under the target that this script generated."""
    target = root / TARGET_DIR
    if not target.is_dir():
        return []
    return sorted(
        path
        for path in target.glob("*.toml")
        if path.read_text(encoding="utf-8").startswith("# Generated from ")
    )


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    """Write or check ``.codex/agents/*.toml``; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when stale")
    args = parser.parse_args(argv)
    outputs = build(root)
    orphans = [path for path in generated_files(root) if path not in outputs]
    stale = [
        path
        for path, text in outputs.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    if args.check:
        for path in stale:
            print(f"stale: {path.relative_to(root).as_posix()}")
        for path in orphans:
            print(f"orphan: {path.relative_to(root).as_posix()}")
        if stale or orphans:
            print("run: uv run python scripts/sync_agents.py")
            return 1
        print(f"agents: {len(outputs)} Codex agents up to date")
        return 0
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(outputs[path], encoding="utf-8")
        print(f"wrote {path.relative_to(root).as_posix()}")
    for path in orphans:
        path.unlink()
        print(f"removed {path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
