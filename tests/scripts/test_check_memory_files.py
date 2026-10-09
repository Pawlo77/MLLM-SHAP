"""Tests for the memory and vault content guard.

@see [[scripts/hooks/check_memory_files.py]]
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from _vault_helpers import PYPROJECT, write


@pytest.fixture(scope="module")
def guard(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/hooks/check_memory_files.py."""
    return script("check_memory_files")


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("api_key = abcdefgh12345678", "credential assignment"),
        ("password: hunter2hunter2", "credential assignment"),
        ("key sk-" + "A" * 24, "OpenAI-style secret key"),
        ("token ghp_" + "b" * 36, "GitHub personal access token"),
        ("-----BEGIN " + "RSA PRIV" + "ATE KEY-----", "PEM block"),
        ("Authorization: Bearer " + "c" * 30, "bearer token"),
        ("mail someone@private-company.com", "e-mail address"),
        ("server at 10.1.2.3", "non-loopback IPv4 address"),
    ],
)
def test_each_rule_has_a_positive(guard: ModuleType, line: str, rule: str) -> None:
    assert any(rule in found for found in guard.check_line(line, set()))


@pytest.mark.parametrize(
    "line",
    [
        "The maintainer is maintainer@example.org.",
        "Reviewed-by: Agent <noreply@github.com>",
        "Local server http://127.0.0.1:1234/v1",
        "Version 0.1.16 and 2026-10-07 are not addresses.",
        "A token is a whitespace-delimited player.",
    ],
)
def test_clean_lines_pass(guard: ModuleType, line: str) -> None:
    assert guard.check_line(line, {"maintainer@example.org"}) == []


def test_run_reports_path_and_line(guard: ModuleType, tmp_path: Path) -> None:
    write(tmp_path, "pyproject.toml", PYPROJECT)
    clean = write(tmp_path, "knowledge/clean.md", "Nothing to see.\n")
    dirty = write(tmp_path, ".remember/today.md", "ok\npassword = supersecret1\n")
    findings = guard.run([clean, dirty], root=tmp_path)
    assert findings == [".remember/today.md:2: credential assignment"]


def test_maintainer_emails_come_from_pyproject(
    guard: ModuleType, tmp_path: Path
) -> None:
    write(tmp_path, "pyproject.toml", PYPROJECT)
    assert "maintainer@example.org" in guard.maintainer_emails(tmp_path)


def test_git_identity_is_not_allowed(guard: ModuleType, tmp_path: Path) -> None:
    """A work address set as git user.email still fails in a .remember/ file."""
    work = "pat.doe@acme-games.test"
    write(tmp_path, "pyproject.toml", PYPROJECT)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", work], cwd=tmp_path, check=True)
    assert work not in guard.maintainer_emails(tmp_path)
    memory = write(tmp_path, ".remember/today.md", f"Asked {work} about it.\n")
    findings = guard.run([memory], root=tmp_path)
    assert findings == [f".remember/today.md:1: e-mail address {work}"]
