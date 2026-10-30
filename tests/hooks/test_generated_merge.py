"""Two branches that change a generated note merge without a conflict.

The rule is the ``knowledge/generated/**`` line of ``.gitattributes``.
"""

import shutil
from pathlib import Path

from tests.hooks._hookrun import ROOT, git


def test_generated_notes_merge_without_conflict(tmp_path: Path) -> None:
    """Both sides add a row to the same note: the merge still completes."""
    git(tmp_path, "init", "-q", "-b", "main")
    shutil.copy(ROOT / ".gitattributes", tmp_path / ".gitattributes")
    note = tmp_path / "knowledge" / "generated" / "Modules.md"
    note.parent.mkdir(parents=True)
    note.write_text("| Module |\n|---|\n| base |\n", encoding="utf-8")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "chore: base")
    for branch in ("a", "b"):
        git(tmp_path, "checkout", "-q", "-b", branch, "main")
        note.write_text(
            f"| Module |\n|---|\n| base |\n| {branch} |\n", encoding="utf-8"
        )
        git(tmp_path, "commit", "-qam", f"feat: add {branch}")
    git(tmp_path, "checkout", "-q", "a")
    git(tmp_path, "merge", "-q", "--no-edit", "b")
    assert "| b |" in note.read_text(encoding="utf-8")
