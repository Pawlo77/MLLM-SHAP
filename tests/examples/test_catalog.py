"""Ensure every examples/*.ipynb is cataloged in gallery docs."""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "examples"
_CATALOGS = (
    _EXAMPLES / "README.md",
    _ROOT / "docs" / "examples.rst",
    _ROOT / "README.md",
)


def _notebook_names() -> list[str]:
    return sorted(p.name for p in _EXAMPLES.glob("*.ipynb"))


@pytest.mark.parametrize("catalog", _CATALOGS, ids=lambda p: p.name)
def test_every_notebook_listed_in_catalog(catalog: Path) -> None:
    """Each examples/*.ipynb name appears in README, docs/examples.rst, root README."""
    text = catalog.read_text(encoding="utf-8")
    missing = [name for name in _notebook_names() if name not in text]
    assert not missing, f"{catalog.relative_to(_ROOT)} missing: {missing}"


def test_catalog_has_at_least_one_notebook() -> None:
    """Guard against empty examples/ silently passing catalog checks."""
    assert _notebook_names(), "examples/ has no .ipynb files"
