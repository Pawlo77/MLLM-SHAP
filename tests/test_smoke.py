"""Smoke tests for the mllm_shap package.

@see [[knowledge/architecture/core]]
"""

from importlib.metadata import version
from pathlib import Path

import mllm_shap


def test_version_is_defined() -> None:
    """Package exposes the installed distribution version."""
    assert mllm_shap.__version__ == version("mllm-shap")


def test_examples_scaffold_exists() -> None:
    """Examples gallery README exists for future notebooks."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples" / "README.md").is_file()
