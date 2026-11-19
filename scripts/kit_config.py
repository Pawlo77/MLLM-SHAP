"""Project settings of the shared agent tooling (``[tool.agent-kit]`` of pyproject).

The template owns the engines under ``scripts/``; what differs per repository (vault
folders, tree folders, review paths, budgets) lives in that table. Stdlib only.
"""

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

TOOLS = ("claude", "cursor", "codex")
"""Agent tools the template can configure (Copier answer ``agent_tools``)."""


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


def agent_tools(root: Path = ROOT) -> frozenset[str]:
    """Return the agent tools answered in ``.copier-answers.yml``.

    A missing file or answer means all of :data:`TOOLS` (projects from before the
    question keep every tool until their next ``copier update`` records the answer).
    """
    try:
        lines = (root / ".copier-answers.yml").read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset(TOOLS)
    for index, line in enumerate(lines):
        key, _, value = line.partition(":")
        if key != "agent_tools":
            continue
        if value.strip():  # flow style: [claude, codex] or []
            items = value.strip().strip("[]").split(",")
        else:  # block style: one "- name" line per tool
            items = []
            for item in lines[index + 1 :]:
                if not item.lstrip().startswith("- "):
                    break
                items.append(item.lstrip()[2:])
        return frozenset(i.strip().strip("'\"") for i in items) & frozenset(TOOLS)
    return frozenset(TOOLS)


def answer(root: Path, key: str) -> str | None:
    """Return the scalar Copier answer ``key`` (``None`` when missing or unreadable)."""
    try:
        lines = (root / ".copier-answers.yml").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, value = line.partition(":")
        if name == key:
            return value.strip().strip("'\"")
    return None
