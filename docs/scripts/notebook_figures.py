"""Extract figures of example notebooks for the guides.

A code cell tagged ``docs-figure:<name>`` in ``examples/*.ipynb`` gives its last
PNG output to ``docs/_static/notebook_figures/<name>.png``, which a guide shows
with ``.. figure::``. The notebooks hold real-model outputs that the docs build
cannot recompute (no model downloads in the build).

Usage::

    uv run python docs/scripts/notebook_figures.py
"""

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "_static" / "notebook_figures"
TAG = "docs-figure:"


def _tagged(notebook: Path) -> list[tuple[str, dict[str, object]]]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return [
        (tag.removeprefix(TAG), cell)
        for cell in cells
        for tag in cell.get("metadata", {}).get("tags", [])
        if tag.startswith(TAG)
    ]


def tagged_names(notebooks: list[Path]) -> set[str]:
    """Return the figure names the notebooks tag."""
    return {name for notebook in notebooks for name, _ in _tagged(notebook)}


def extract(notebooks: list[Path], output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Write each tagged cell's last PNG to ``output_dir``; return the files.

    Raises:
        ValueError: If a tagged cell has no PNG output (run ``make notebooks``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for notebook in notebooks:
        for name, cell in _tagged(notebook):
            images = [
                output["data"]["image/png"]
                for output in cell.get("outputs", [])
                if "image/png" in output.get("data", {})
            ]
            if not images:
                msg = f"{notebook.name}: cell {TAG}{name} has no PNG output"
                raise ValueError(msg)
            path = output_dir / f"{name}.png"
            path.write_bytes(base64.b64decode(images[-1]))
            written.append(path)
    return written


def main() -> int:
    """Extract the figures of every example notebook."""
    try:
        written = extract(sorted((ROOT / "examples").glob("*.ipynb")))
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
