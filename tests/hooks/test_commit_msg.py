"""Tests for conventional commit message validation.

@see [[scripts/hooks/commit_msg.py]]
"""

import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / "scripts" / "hooks" / "commit_msg.py"


@pytest.fixture(scope="module")
def commit_msg():
    """Load commit_msg.py without installing it as a package."""
    spec = importlib.util.spec_from_file_location("commit_msg", _HOOK)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "message",
    [
        "feat: add backend bench harness",
        "fix: correct mask codec edge case",
        "docs: add repository URL to README",
        "test: cover catalog drift",
        "refactor: split python quality rules",
        "chore: align make check with agent-check",
        "ci: filter lms gpu bench markers",
        "build: bump uv required version",
        "perf: stream coalition enumeration",
        "feat(runtime): bound scheduler memory",
        "[prek] auto fixes from pre-commit hooks",
        "[prek] pre-commit autoupdate",
        "feat: add x\n\nBody with details.",
        "Merge branch 'main' into feat/x",
        "Merge remote-tracking branch 'origin/main'",
        "Merge pull request #12 from owner/fix/x",
        'Revert "feat: add x"',
    ],
)
def test_valid_commit_messages(commit_msg, message: str) -> None:
    """Accepted subjects match git-commits.mdc / prek autofix."""
    assert commit_msg.validate_commit_message(message) is None


@pytest.mark.parametrize(
    "message",
    [
        "",
        "WIP",
        "Feat: bad capital type",
        "feat: trailing period.",
        "feat add missing colon",
        "unknown: not a type",
        "Merge stuff",
        "revert things",
        "feat: " + ("x" * 80),
        "feat: add x\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
        "feat: add x\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)",
    ],
)
def test_invalid_commit_messages(commit_msg, message: str) -> None:
    """Rejected subjects return an error string."""
    assert commit_msg.validate_commit_message(message) is not None
