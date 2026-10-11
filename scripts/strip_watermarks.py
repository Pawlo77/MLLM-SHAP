"""Strip AI attribution watermarks from files and commit messages.

Usage:
    python scripts/strip_watermarks.py [--check] FILE ...

Removes whole lines that are only an AI attribution: ``Co-Authored-By``-style
trailers naming an assistant, ``Generated with <assistant>`` banners and ``written by
<assistant>`` comments. Prints ``path:line: <text>`` per removed line; ``--check``
only reports and exits 1 on a finding. Run by prek, the edit hooks and skill ``ship``.

@domain tooling
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP = ("scripts/strip_watermarks.py", "tests/scripts/test_strip_watermarks.py")
"""Repository paths that quote the patterns on purpose."""

_AI = (
    r"(claude|anthropic|copilot|codex|openai|chatgpt|gpt-?[0-9o]|gemini|cursor|devin|"
    r"aider|windsurf|an ai\b|ai assistant|ai agent|llm\b)"
)
_AI_TRAILER = (
    rf"(?:{_AI}|grok|deepseek|jules|mistral|qwen|llama|kimi|cline|junie|kiro|augment|"
    r"sourcegraph|\[bot\]|noreply@(?:anthropic|openai|cursor|x)\.)"
)
"""Names accepted after a trailer key, where a person's name is not expected."""
_VERB = r"(generated|written|created|made|drafted|authored|produced|co-authored)"
WATERMARK = re.compile(
    rf"^(?:(?:co-authored|assisted|generated|authored|signed-off|ai-generated|made)"
    rf"-(?:by|with)\s*:.*{_AI_TRAILER}"
    rf"|(?:🤖\s*)?{_VERB}\s+(?:with|by|using)\s+(?:the\s+)?(?:help\s+of\s+)?\[?{_AI}"
    rf"|(?:this|the)\s+\w+\s+(?:was|is)\s+{_VERB}\s+(?:with|by|using)\s+\[?{_AI})",
    re.IGNORECASE,
)
"""Matches a normalised line that is nothing but an AI attribution."""

_LEAD = re.compile(r"""^(?:#+|//+|/\*+|\*+|<!--|--|;+|%+|["']|-\s|>)\s*""")
_TAIL = re.compile(r"""(?:\\n)?["']?,?\s*(?:-->|\*/)?\s*$""")


def is_watermark(line: str) -> bool:
    """Return True when ``line`` minus comment markers is an AI attribution."""
    text = line.strip()
    for _ in range(2):
        text = _LEAD.sub("", text)
    text = _TAIL.sub("", text)
    return bool(WATERMARK.match(text))


def strip_text(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Return ``text`` without watermark lines and the removed ``(line_no, line)``."""
    kept: list[str] = []
    removed: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(keepends=True), start=1):
        if is_watermark(line):
            removed.append((number, line.strip()))
        else:
            kept.append(line)
    if not removed:
        return text, []
    result = "".join(kept)
    if result.strip():
        result = result.rstrip("\n") + "\n"
    return result, removed


def _readable(path: Path) -> str | None:
    rel = (
        path.resolve().relative_to(ROOT).as_posix()
        if path.resolve().is_relative_to(ROOT)
        else ""
    )
    if rel in SKIP or not path.is_file():
        return None
    data = path.read_bytes()
    if b"\0" in data[:8000]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def strip_file(path: Path, *, check: bool = False) -> list[tuple[int, str]]:
    """Strip ``path`` in place (report only with ``check``); return the removals."""
    text = _readable(path)
    if text is None:
        return []
    new_text, removed = strip_text(text)
    if removed and not check:
        path.write_text(new_text, encoding="utf-8")
    return removed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    check = "--check" in args
    paths = [Path(a) for a in args if a != "--check"]
    if not paths:
        print("usage: strip_watermarks.py [--check] FILE ...", file=sys.stderr)
        return 2
    found = 0
    for path in paths:
        for number, line in strip_file(path, check=check):
            found += 1
            print(f"{path}:{number}: {line}")
    return 1 if check and found else 0


if __name__ == "__main__":
    raise SystemExit(main())
