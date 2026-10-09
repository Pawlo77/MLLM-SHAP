"""Generate Claude Code rules (``.claude/rules/``) from ``.cursor/rules/*.mdc``.

Each scoped rule becomes ``<name>.md`` with its globs as ``paths``; every
``alwaysApply: true`` rule is merged into ``always.md``. Bodies are copied verbatim.

Usage::

    uv run python scripts/sync_rules.py           # write .claude/rules/
    uv run python scripts/sync_rules.py --check   # exit 1 when outputs are stale
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

SOURCE_DIR = ROOT / ".cursor" / "rules"
"""Cursor rules, the single source."""

TARGET_DIR = ROOT / ".claude" / "rules"
"""Generated Claude Code rules."""

ALWAYS_NAME = "always.md"
"""Output file that merges every always-apply rule."""

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
    """Return the generated files as ``{file name: content}``."""
    outputs: dict[str, str] = {}
    always: list[str] = []
    for path in sorted(sources):
        fields, body = split_frontmatter(path.read_text(encoding="utf-8"))
        source = path.relative_to(ROOT).as_posix()
        header = HEADER.format(source=source)
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


def main(argv: list[str] | None = None) -> int:
    """Write or check ``.claude/rules/``; return the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="exit 1 when outputs are stale"
    )
    args = parser.parse_args(argv)

    outputs = render(list(SOURCE_DIR.glob("*.mdc")))
    existing = (
        {p.name for p in TARGET_DIR.glob("*.md")} if TARGET_DIR.is_dir() else set()
    )
    stale = sorted(
        name
        for name, content in outputs.items()
        if not (TARGET_DIR / name).is_file()
        or (TARGET_DIR / name).read_text(encoding="utf-8") != content
    )
    extra = sorted(existing - outputs.keys())

    if args.check:
        for name in stale:
            print(f"stale: .claude/rules/{name}")
        for name in extra:
            print(f"not generated: .claude/rules/{name}")
        if stale or extra:
            print("run: uv run python scripts/sync_rules.py", file=sys.stderr)
            return 1
        return 0

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for name in stale:
        (TARGET_DIR / name).write_text(outputs[name], encoding="utf-8")
        print(f"wrote .claude/rules/{name}")
    for name in extra:
        (TARGET_DIR / name).unlink()
        print(f"removed .claude/rules/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
