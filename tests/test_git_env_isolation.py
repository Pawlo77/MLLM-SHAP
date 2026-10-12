"""Tests start without the git hook environment.

@see [[tests/AGENTS.md]]
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = (
    "import os\n\n"
    "def test_probe():\n"
    "    assert not [key for key in os.environ if key.startswith('GIT_')]\n"
)


def test_git_env_is_hidden_from_tests() -> None:
    """A pytest run under a hook's GIT_* variables sees none of them."""
    probe = ROOT / "tests" / "probe_git_env.py"  # inside tests/ so conftest applies
    probe.write_text(PROBE, encoding="utf-8")
    env = {**os.environ, "GIT_DIR": "/nonexistent", "GIT_INDEX_FILE": "/nonexistent"}
    try:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                str(probe),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        probe.unlink()
    assert run.returncode == 0, run.stdout
