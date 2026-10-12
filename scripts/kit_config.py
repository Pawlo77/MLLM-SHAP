"""Project settings of the shared agent tooling (``[tool.agent-kit]`` of pyproject).

The template owns the engines under ``scripts/``; what differs per repository (vault
folders, tree folders, review paths, budgets) lives in that table. Stdlib only.
"""

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(root: Path = ROOT) -> dict[str, Any]:
    """Return the ``[tool.agent-kit]`` table (empty when missing or unreadable)."""
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    table = data.get("tool", {}).get("agent-kit", {})
    return table if isinstance(table, dict) else {}


def words(root: Path, key: str, default: str) -> tuple[str, ...]:
    """Return the whitespace-separated words of string setting ``key``."""
    return tuple(str(load(root).get(key, default)).split())
