"""Tests for conventional commit message validation."""

import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / ".cursor" / "hooks" / "commit_msg.py"


@pytest.fixture(scope="module")
def commit_msg():
    """Load commit_msg.py without installing it as a package."""
    spec = importlib.util.spec_from_file_location("commit_msg", _HOOK)
    assert spec is not None and spec.loader is not None
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
        "feat: " + ("x" * 80),
    ],
)
def test_invalid_commit_messages(commit_msg, message: str) -> None:
    """Rejected subjects return an error string."""
    assert commit_msg.validate_commit_message(message) is not None
