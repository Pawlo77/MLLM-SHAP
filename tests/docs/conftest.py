"""Make the Sphinx extensions and docs scripts importable in the docs tests.

@see [[docs/_ext/public_exports.py]]
"""

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs"

for directory in (DOCS / "_ext", DOCS / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
