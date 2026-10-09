"""Smoke test that the base wheel does not pull heavy ML dependencies.

Builds a wheel and resolves it in an isolated environment (network), so it is
marked ``slow`` and runs in ``make tests-all`` only.

@see [[pyproject.toml]]
"""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_base_wheel_import_has_no_torch() -> None:
    """Isolated wheel install does not include torch."""
    root = Path(__file__).resolve().parents[1]
    wheels = sorted(root.glob("dist/nlp_shap-*.whl"))
    if not wheels:
        subprocess.run(["uv", "build"], cwd=root, check=True)
        wheels = sorted(root.glob("dist/nlp_shap-*.whl"))
    wheel = wheels[-1]
    script = (
        "import importlib.util; "
        "import nlp_shap; "
        "assert importlib.util.find_spec('torch') is None; "
        "print(nlp_shap.__version__)"
    )
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            str(wheel),
            "python",
            "-c",
            script,
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
