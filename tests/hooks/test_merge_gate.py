"""Merge gate: ``gh pr merge`` needs a routine marker for the session.

@see [[scripts/hooks/merge_gate.sh]]
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import pr_policy
from tests.hooks._hookrun import CLAUDE, CURSOR, HAVE_TOOLS, ROOT, git, scratch_repo

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="needs bash and jq")

MERGE = "gh pr merge 7 --merge --delete-branch"
FAKE_UV = (
    "#!/bin/bash\nwhile [ $1 != python ]; do shift; done; shift\nexec python3 $@\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Scratch repo on a ``docs:`` branch with pr_policy.py; a fake uv sits in bin/."""
    root = scratch_repo(tmp_path)
    for name in ("pr_policy.py", "kit_config.py"):
        shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
    (root / ".gitignore").write_text(".remember/tmp/\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "chore: policy")
    git(root, "switch", "-qc", "docs/x")
    (root / "NOTE.md").write_text("note\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "docs: note")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "uv").write_text(FAKE_UV, encoding="utf-8")
    (tmp_path / "bin" / "uv").chmod(0o755)
    return root


def _gate(repo: Path, command: str, *, cursor: bool = False) -> dict[str, object]:
    payload = {**CURSOR, "command": command}
    if not cursor:
        payload = {**CLAUDE, "tool_input": {"command": command}}
    path = f"{repo.parent / 'bin'}{os.pathsep}{os.environ['PATH']}"
    proc = subprocess.run(
        ["bash", str(repo / "scripts" / "hooks" / "merge_gate.sh")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=repo,
        env={**os.environ, "PATH": path},
        check=False,
    )
    return json.loads(proc.stdout) if proc.stdout else {}


def test_other_commands_pass(repo: Path) -> None:
    """A non-merge command: no output for Claude, an allow for Cursor."""
    assert _gate(repo, "ls") == {}
    assert _gate(repo, "ls", cursor=True) == {"permission": "allow"}


def test_merge_denied_without_marker(repo: Path) -> None:
    """No marker for the session: both shapes are denied with a reason."""
    claude = _gate(repo, MERGE)["hookSpecificOutput"]
    assert claude["permissionDecision"] == "deny"
    assert "no current marker" in claude["permissionDecisionReason"]
    assert _gate(repo, MERGE, cursor=True)["permission"] == "deny"


def test_merge_allowed_with_routine_marker(repo: Path) -> None:
    """A routine marker with PR 7, made on HEAD, lets the merge through."""
    for key in ("s1", "c1"):  # the session keys of CLAUDE and CURSOR
        pr_policy.write_marker("main", key, repo, pr=7)
    assert _gate(repo, MERGE) == {}
    assert _gate(repo, MERGE, cursor=True) == {"permission": "allow"}
