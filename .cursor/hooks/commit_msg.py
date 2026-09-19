"""Conventional commit subject validation shared by prek and Cursor hooks."""

import re
import sys
from pathlib import Path

COMMIT_TYPES: tuple[str, ...] = (
    "feat",
    "fix",
    "docs",
    "test",
    "refactor",
    "chore",
    "ci",
    "build",
    "perf",
)

_TYPE_ALT = "|".join(COMMIT_TYPES)
_SUBJECT_RE = re.compile(rf"^({_TYPE_ALT})(\([a-zA-Z0-9/_-]+\))?: .+$")

# prek autofix / autoupdate commit messages from .pre-commit-config.yaml
_PREK_PREFIX = "[prek]"


def subject_line(message: str) -> str:
    """Return the first non-empty line of a commit message."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def validate_commit_message(message: str) -> str | None:
    """Return an error string if ``message`` is invalid, else ``None``."""
    subject = subject_line(message)
    if not subject:
        return "commit message is empty"
    if subject.startswith(_PREK_PREFIX):
        return None
    if subject.endswith("."):
        return "subject must not end with a period"
    if len(subject) > 72:
        return f"subject too long ({len(subject)} > 72 chars)"
    if not _SUBJECT_RE.match(subject):
        types = "|".join(COMMIT_TYPES)
        return (
            f"subject must match 'type: summary' with type in ({types}); "
            f"got {subject!r}"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI: prek commit-msg (path arg) or ``--check-text`` for tests."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--check-text":
        text = args[1] if len(args) > 1 else ""
        err = validate_commit_message(text)
        if err:
            print(err, file=sys.stderr)
            return 1
        return 0

    if not args:
        print("usage: commit_msg.py <COMMIT_EDITMSG>|--check-text MSG", file=sys.stderr)
        return 2

    path = Path(args[0])
    text = path.read_text(encoding="utf-8")
    err = validate_commit_message(text)
    if err:
        print(f"conventional commit check failed: {err}", file=sys.stderr)
        print(
            "format: type: summary  (types: "
            + ", ".join(COMMIT_TYPES)
            + "; no trailing period; ≤72 chars)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
