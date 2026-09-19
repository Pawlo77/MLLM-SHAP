"""Tests for Cursor deny-shell hook logic."""

import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / ".cursor" / "hooks" / "deny_shell.py"


@pytest.fixture(scope="module")
def deny_shell():
    """Load deny_shell.py without installing it as a package."""
    spec = importlib.util.spec_from_file_location("deny_shell", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("cmd", "permission"),
    [
        ("pip install foo", "deny"),
        ("pip3 install foo", "deny"),
        ("python -m pip install foo", "deny"),
        ("conda install numpy", "deny"),
        ("mamba install numpy", "deny"),
        ("pipx install cowsay", "deny"),
        ("pre-commit run --all-files", "deny"),
        ("git push --force origin main", "deny"),
        ("git push -f origin HEAD", "deny"),
        ("git push --force-with-lease origin main", "deny"),
        ("git commit -m 'feat: add x'", "allow"),
        ("git push origin main", "deny"),
        ("uv run prek run --all-files", "allow"),
        ("uv sync --all-groups", "allow"),
        ("make install", "allow"),
        ("git push -u origin feature/foo", "allow"),
        ('echo "pip install foo" | cat', "allow"),
        ("uv run pytest tests/", "allow"),
    ],
)
def test_decide_permission(deny_shell, cmd: str, permission: str) -> None:
    """Package and git policies compose into a single decision."""
    assert deny_shell.decide(cmd)["permission"] == permission
