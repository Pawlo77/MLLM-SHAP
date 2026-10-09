"""The vault checker accepts solution notes and requires their sections.

@see [[scripts/check_vault.py]]
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from _vault_helpers import note, write

SECTIONS = (
    "## Problem\n\n- p\n\n## What was tried\n\n- t\n\n"
    "## What worked\n\n- w\n\n## Reusable rule\n\n- r\n"
)


@pytest.fixture(scope="module")
def checker(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/check_vault.py."""
    return script("check_vault")


def _add_solution(root: Path, body: str) -> None:
    write(
        root,
        "knowledge/handbook/Solutions/2026-10-07 - Demo fix.md",
        note(
            "Demo fix",
            "solution",
            body,
            extra='pr: "https://example.org/pr/1"\nplan-task: "plans/x.md, Task 1.1"\n',
        ),
    )
    write(
        root,
        "knowledge/Map.md",
        (root / "knowledge" / "Map.md").read_text() + "[[2026-10-07 - Demo fix]]\n",
    )


def _errors(checker: ModuleType, root: Path) -> list[str]:
    findings, _, _ = checker.run(root)
    return [str(item) for item in checker.iter_errors(findings)]


def test_solution_note_is_valid(checker: ModuleType, clean_vault: Path) -> None:
    """A linked solution note with every required section passes."""
    _add_solution(clean_vault, SECTIONS)
    assert "solution" in checker.NOTE_TYPES
    assert _errors(checker, clean_vault) == []


def test_solution_note_needs_sections(checker: ModuleType, clean_vault: Path) -> None:
    """A solution note without "What was tried" is reported."""
    _add_solution(clean_vault, SECTIONS.replace("## What was tried\n\n- t\n\n", ""))
    errors = _errors(checker, clean_vault)
    assert any("missing section '## What was tried'" in item for item in errors)


def test_watch_folder_is_skipped(
    checker: ModuleType, script: Callable[[str], ModuleType], clean_vault: Path
) -> None:
    """A literature-watch shortlist (unlinked, git-ignored) is not a vault note."""
    write(
        clean_vault,
        "knowledge/papers/_watch/2026-10-07.md",
        '---\ntitle: "Watch"\ntype: handbook\n---\n\nUnlinked; [[Missing note]].\n',
    )
    assert _errors(checker, clean_vault) == []
    builder = script("build_knowledge")
    assert all(
        "_watch" not in path.parts
        for path, _ in builder.notes_of_type(clean_vault, "handbook")
    )
