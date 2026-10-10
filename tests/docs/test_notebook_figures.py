"""Figures of example notebooks extracted for the guides.

@see [[docs/scripts/notebook_figures.py]]
"""

import base64
import json
from pathlib import Path

import notebook_figures
import pytest

PNG = b"\x89PNG\r\n\x1a\nfake"


def _notebook(path: Path, cells: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4}))
    return path


def _cell(tags: list[str], *, image: bool = True) -> dict[str, object]:
    data = {"image/png": base64.b64encode(PNG).decode()} if image else {}
    outputs = [{"output_type": "display_data", "data": data, "metadata": {}}]
    return {"cell_type": "code", "metadata": {"tags": tags}, "outputs": outputs}


def test_tagged_cells_become_named_pngs(tmp_path: Path) -> None:
    notebook = _notebook(
        tmp_path / "demo.ipynb",
        [_cell(["docs-figure:refinement"]), _cell([]), _cell(["other"])],
    )
    written = notebook_figures.extract([notebook], tmp_path / "out")
    assert written == [tmp_path / "out" / "refinement.png"]
    assert written[0].read_bytes() == PNG


def test_a_tagged_cell_without_an_image_fails(tmp_path: Path) -> None:
    notebook = _notebook(
        tmp_path / "demo.ipynb", [_cell(["docs-figure:empty"], image=False)]
    )
    with pytest.raises(ValueError, match="docs-figure:empty"):
        notebook_figures.extract([notebook], tmp_path / "out")


def test_every_guide_figure_comes_from_a_tagged_notebook_cell() -> None:
    """Each ``_static/notebook_figures/<name>.png`` a page uses is extracted."""
    root = Path(__file__).resolve().parents[2]
    names = notebook_figures.tagged_names(sorted((root / "examples").glob("*.ipynb")))
    used = {
        line.split("notebook_figures/")[1].removesuffix(".png").strip()
        for page in (root / "docs").rglob("*.rst")
        for line in page.read_text(encoding="utf-8").splitlines()
        if "_static/notebook_figures/" in line
    }
    assert used
    assert used <= names
