"""Smoke tests for the nlp_shap package.

@see [[knowledge/architecture/core]]
"""

from importlib.metadata import version
from pathlib import Path

import nlp_shap


def test_version_is_defined() -> None:
    """Package exposes the installed distribution version."""
    assert nlp_shap.__version__ == version("nlp-shap")


def test_examples_scaffold_exists() -> None:
    """Examples gallery README exists for future notebooks."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "examples" / "README.md").is_file()
