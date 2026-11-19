"""Generate the agent rules from the tool-neutral sources ``.agents/rules/*.mdc``.

For Claude Code (answer ``agent_tools``), each scoped rule becomes
``.claude/rules/<name>.md`` with its globs as ``paths``, and every ``alwaysApply: true``
rule is merged into ``always.md``. For Cursor, each source is copied to
``.cursor/rules/<name>.mdc`` with a generated-file header. Bodies are copied verbatim;
outputs of tools that are not selected are removed.

Usage: ``uv run python scripts/sync_rules.py [--check]``; ``--check`` exits 1 if stale.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import agent_tools

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

SOURCE_DIR = Path(".agents") / "rules"
"""Rule sources in Cursor's ``.mdc`` format, relative to the root."""

CLAUDE_DIR = Path(".claude") / "rules"
"""Generated Claude Code rules."""

CURSOR_DIR = Path(".cursor") / "rules"
"""Generated Cursor rules."""

ALWAYS_NAME = "always.md"
"""Output file that merges every always-apply rule."""

PLUGIN_RULES = frozenset({"ponytail"})
"""Rules a Claude Code plugin injects itself; they get no Claude copy."""

HEADER = "<!-- Generated from {source} by scripts/sync_rules.py; do not edit. -->"
"""First body line of every generated file."""


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split an ``.mdc`` file into flat frontmatter fields and the body.

    Values are kept as raw strings: Cursor globs such as ``**/*.py`` are not valid
    plain YAML scalars, so the frontmatter is parsed line by line.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    fields: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :]).strip("\n")
            return fields, body + "\n"
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip().strip("\"'")
    return {}, text


def globs_of(fields: dict[str, str]) -> list[str]:
    """Return the rule's globs as a list (Cursor stores them comma-separated)."""
    raw = fields.get("globs", "").strip().strip("[]")
    return [g.strip().strip("\"'") for g in raw.split(",") if g.strip()]


def render(sources: list[Path]) -> dict[str, str]:
    """Return the generated Claude Code rules as ``{file name: content}``."""
    outputs: dict[str, str] = {}
    always: list[str] = []
    for path in sorted(sources):
        fields, body = split_frontmatter(path.read_text(encoding="utf-8"))
        header = HEADER.format(source=(SOURCE_DIR / path.name).as_posix())
        if path.stem in PLUGIN_RULES:
            continue
        if fields.get("alwaysApply", "").lower() == "true":
            always.append(f"{header}\n\n{body}")
            continue
        globs = globs_of(fields)
        if not globs:
            # Cursor applies such rules only on request; Claude has no equivalent.
            continue
        paths = "\n".join(f'  - "{g}"' for g in globs)
        outputs[f"{path.stem}.md"] = f"---\npaths:\n{paths}\n---\n\n{header}\n\n{body}"
    if always:
        outputs[ALWAYS_NAME] = "\n".join(always)
    return outputs


def render_cursor(sources: list[Path]) -> dict[str, str]:
    """Return the generated Cursor rules: each source with a header after its fields."""
    outputs: dict[str, str] = {}
    for path in sorted(sources):
        text = path.read_text(encoding="utf-8")
        header = HEADER.format(source=(SOURCE_DIR / path.name).as_posix())
        head, sep, body = text.partition("\n---\n")
        if text.startswith("---\n") and sep:
            outputs[path.name] = f"{head}{sep}\n{header}\n{body}"
        else:
            outputs[path.name] = f"{header}\n\n{text}"
    return outputs


def targets(root: Path) -> list[tuple[Path, str, dict[str, str]]]:
    """Return ``(directory, glob, outputs)`` per tool; unselected tools get none."""
    sources = sorted((root / SOURCE_DIR).glob("*.mdc"))
    tools = agent_tools(root)
    claude = render(sources) if "claude" in tools else {}
    cursor = render_cursor(sources) if "cursor" in tools else {}
    return [(root / CLAUDE_DIR, "*.md", claude), (root / CURSOR_DIR, "*.mdc", cursor)]


def replaceable(path: Path, root: Path) -> bool:
    """Whether ``path`` may be overwritten: generated here, or a verbatim source copy.

    Before ``.agents/rules/`` the template shipped the rules in ``.cursor/rules/``; an
    unedited copy equals its source, an edited one or a project rule does not.
    """
    text = path.read_text(encoding="utf-8")
    source = root / SOURCE_DIR / path.name
    if source.is_file() and source.read_text(encoding="utf-8") == text:
        return True
    body = text.split("\n---\n", 1)[-1]
    return "by scripts/sync_rules.py; do not edit." in body[:400]


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    """Write or check the generated rules; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when outputs are stale"
    )
    args = parser.parse_args(argv)
    stale: list[tuple[Path, str]] = []
    extra: list[Path] = []
    legacy: list[Path] = []
    for folder, pattern, outputs in targets(root):
        existing = sorted(folder.glob(pattern)) if folder.is_dir() else []
        # A hand-written Cursor rule predates .agents/rules/: report, never replace it.
        mine = [p for p in existing if pattern == "*.mdc" and not replaceable(p, root)]
        legacy += mine
        stale += [
            (folder / name, text)
            for name, text in outputs.items()
            if folder / name not in mine
            and (
                not (folder / name).is_file()
                or (folder / name).read_text(encoding="utf-8") != text
            )
        ]
        extra += [p for p in existing if p.name not in outputs and p not in mine]
    for path in legacy:
        rel = path.relative_to(root).as_posix()
        print(f"hand-written: {rel}; merge it into {SOURCE_DIR.as_posix()}/, delete it")
    if args.check:
        for path, _ in stale:
            print(f"stale: {path.relative_to(root).as_posix()}")
        for path in extra:
            print(f"not generated: {path.relative_to(root).as_posix()}")
        if stale or extra or legacy:
            print("run: uv run python scripts/sync_rules.py", file=sys.stderr)
        return 1 if stale or extra or legacy else 0
    for path, text in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(root).as_posix()}")
    for path in extra:
        path.unlink()
        print(f"removed {path.relative_to(root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
