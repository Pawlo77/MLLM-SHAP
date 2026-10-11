"""Tests for the vault-to-Sphinx exporter.

@see [[scripts/export_knowledge.py]]
"""

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from _vault_helpers import note, write


@pytest.fixture(scope="module")
def exporter(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/export_knowledge.py."""
    return script("export_knowledge")


def test_export_rewrites_obsidian_syntax(exporter: ModuleType, tmp_path: Path) -> None:
    write(
        tmp_path,
        "knowledge/concepts/Coalition.md",
        note(
            "Coalition", "concept", "A subset. See [[How it works|the walkthrough]].\n"
        ),
    )
    write(
        tmp_path,
        "knowledge/handbook/How it works.md",
        note(
            "How it works",
            "handbook",
            "> [!warning] Careful\n> text\n\n"
            "Uses [[Coalition]] and [[src/demo.py]] and [[C01 - Bug]].\n\n"
            "![[Coalition]]\n\n```mermaid\nflowchart LR\n  a --> b\n```\n\n"
            "```dataview\nLIST\n```\n",
        ),
    )
    write(tmp_path, "knowledge/issues/C01 - Bug.md", note("C01 - Bug", "issue", "x\n"))
    write(tmp_path, "src/demo.py", '"""Demo."""\n')
    out = tmp_path / "docs" / "handbook"
    assert exporter.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    page = (out / "handbook" / "how-it-works.md").read_text()
    assert page.startswith("# How it works\n")
    assert "> **Warning:** Careful" in page
    assert "[Coalition](../concepts/coalition.md)" in page
    assert "https://github.com/Pawlo77/MLLM-SHAP/blob/main/src/demo.py" in page
    assert "knowledge/issues/C01%20-%20Bug.md" in page
    assert "A subset." in page
    assert "```text" in page
    assert "LIST" not in page
    concept = (out / "concepts" / "coalition.md").read_text()
    assert "[the walkthrough](../handbook/how-it-works.md)" in concept
    assert "[[" not in page + concept


def test_export_skips_plan_phase_sections(exporter: ModuleType, tmp_path: Path) -> None:
    """The Roadmap and decision-log sections naming a plan phase stay in the vault;
    links to them point at GitHub, links to kept sections carry the MyST anchor."""
    write(
        tmp_path,
        "knowledge/handbook/Roadmap.md",
        note("Roadmap", "handbook", "Phase 7 next.\n"),
    )
    write(
        tmp_path,
        "knowledge/handbook/Onboarding.md",
        note(
            "Onboarding",
            "handbook",
            "See [[Roadmap]], [[Process decisions#2026-10-07 — One go|one go]] and "
            "[[Process decisions#2026-10-07 — Merge strategy|merges]].\n",
        ),
    )
    write(
        tmp_path,
        "knowledge/decisions/Process decisions.md",
        note(
            "Process decisions",
            "decision-log",
            "Intro.\n\n## 2026-10-07 — One go\n\nPhases 3 and 2 ship together.\n\n"
            "## 2026-10-07 — Merge strategy\n\nMerge commits, no phases.\n",
        ),
    )
    out = tmp_path / "docs" / "handbook"
    assert exporter.main(["--root", str(tmp_path), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.rglob("*.md")) == [
        "onboarding.md",
        "process-decisions.md",
    ]
    log = (out / "decisions" / "process-decisions.md").read_text()
    assert "Merge commits" in log
    assert "Phases" not in log
    page = (out / "handbook" / "onboarding.md").read_text()
    assert "blob/main/knowledge/handbook/Roadmap.md" in page
    assert "[one go](https://github.com/" in page
    assert (
        "[merges](../decisions/process-decisions.md#2026-10-07--merge-strategy)" in page
    )
