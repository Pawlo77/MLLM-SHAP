"""Helpers to run the shared agent hook scripts in tests."""

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOOKS = ROOT / "scripts" / "hooks"

CURSOR = {"cursor_version": "2.5", "conversation_id": "c1"}
CLAUDE = {"session_id": "s1", "hook_event_name": "PreToolUse"}

HAVE_TOOLS = shutil.which("bash") is not None and shutil.which("jq") is not None


def run_hook(
    script: Path, payload: dict[str, object], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` with ``payload`` as JSON on stdin."""
    return subprocess.run(
        ["bash", str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
        timeout=120,
        check=False,
    )


def git(cwd: Path, *args: str) -> None:
    """Run a quiet git command in ``cwd``."""
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def scratch_repo(tmp: Path) -> Path:
    """Create a git repo under ``tmp`` holding a copy of the hook scripts."""
    repo = tmp / "repo"
    (repo / "scripts" / "hooks").mkdir(parents=True)
    for f in HOOKS.glob("*.sh"):
        shutil.copy2(f, repo / "scripts" / "hooks" / f.name)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "mllm-shap"\nversion = "9.8.7"\n', encoding="utf-8"
    )
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "chore: base")
    return repo
