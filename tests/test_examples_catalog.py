"""Ensure every examples/*.ipynb is cataloged and is a notebook-only example.

@see [[examples/README.md]]
"""

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples"
_CATALOGS = (
    _EXAMPLES / "README.md",
    _ROOT / "docs" / "examples.rst",
)


def _notebook_names() -> list[str]:
    return sorted(p.name for p in _EXAMPLES.glob("*.ipynb"))


@pytest.mark.parametrize("catalog", _CATALOGS, ids=lambda p: p.name)
def test_every_notebook_listed_in_catalog(catalog: Path) -> None:
    """Each notebook name appears in examples/README.md and docs/examples.rst."""
    text = catalog.read_text(encoding="utf-8")
    missing = [name for name in _notebook_names() if name not in text]
    assert not missing, f"{catalog.relative_to(_ROOT)} missing: {missing}"


def test_catalog_has_at_least_one_notebook() -> None:
    """Guard against empty examples/ silently passing catalog checks."""
    assert _notebook_names(), "examples/ has no .ipynb files"


def test_examples_are_notebooks_only() -> None:
    """Examples ship as notebooks with stored outputs; no script pairs."""
    assert not list(_EXAMPLES.glob("*.py")), "examples/ must hold notebooks only"
    for path in _EXAMPLES.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert "jupytext" not in notebook.get("metadata", {}), path.name
        code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        assert any(c["outputs"] for c in code_cells), f"{path.name} has no outputs"
