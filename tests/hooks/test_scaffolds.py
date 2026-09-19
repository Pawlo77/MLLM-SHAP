"""Dry-run smoke for Cursor scaffold scripts."""

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_new_plugin_dry_run() -> None:
    """Plugin scaffold dry-run exits 0 and prints planned paths."""
    script = (
        _ROOT / ".cursor" / "skills" / "nlp-shap-plugin" / "scripts" / "new_plugin.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script), "value_fns", "scaffold_probe", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "scaffold_probe" in proc.stdout
    assert "module=" in proc.stdout


def test_new_notebook_dry_run() -> None:
    """Notebook scaffold dry-run exits 0 and prints catalog reminder."""
    script = (
        _ROOT
        / ".cursor"
        / "skills"
        / "nlp-shap-notebook"
        / "scripts"
        / "new_notebook.py"
    )
    proc = subprocess.run(
        [sys.executable, str(script), "scaffold_probe", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "scaffold_probe.ipynb" in proc.stdout
    assert "catalog:" in proc.stdout
