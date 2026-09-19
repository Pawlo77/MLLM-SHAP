#!/usr/bin/env python3
"""Scaffold a new examples/ Jupyter notebook with cell-1 import stub."""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
_EXAMPLES = _ROOT / "examples"


def _notebook(title: str, stem: str) -> dict:
    """Return a minimal nbformat-v4 notebook dict."""
    md = (
        f"# {title}\n\n"
        "## Goal\n\n"
        "Describe the public workflow this notebook proves.\n\n"
        "## Business context\n\n"
        "Why stakeholders care (auditability, trust, risk).\n\n"
        "## Prerequisites\n\n"
        "`make install` from repo root. Note optional extras if needed.\n"
    )
    imports = (
        "# Cell 1 — all imports (stdlib / third-party, then nlp_shap)\n"
        "from nlp_shap import ExplainRunner  # replace with symbols you need\n"
    )
    body = (
        "# Demonstrate the workflow here\n"
        "raise NotImplementedError("
        f'"fill in {stem}.ipynb — then make notebooks")\n'
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": md.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": imports.splitlines(keepends=True),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Demo\n",
                    "\n",
                    "Technical behavior + business motivation.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": body.splitlines(keepends=True),
            },
        ],
    }


def main() -> int:
    """Create examples/<stem>.ipynb and remind about catalog updates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stem", help="notebook filename without .ipynb")
    parser.add_argument(
        "--title",
        default="",
        help="notebook H1 title (default: stem as title case)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stem = args.stem.removesuffix(".ipynb")
    title = args.title or stem.replace("_", " ").title()
    path = _EXAMPLES / f"{stem}.ipynb"

    print(f"notebook={path.relative_to(_ROOT)}")
    print("catalog: examples/README.md, docs/examples.rst, README.md")

    if args.dry_run:
        return 0
    if path.exists():
        print(f"error: already exists: {path}", file=sys.stderr)
        return 1

    path.write_text(
        json.dumps(_notebook(title, stem), indent=1) + "\n", encoding="utf-8"
    )
    print("next: implement cells, update catalogs, make notebooks, make agent-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
