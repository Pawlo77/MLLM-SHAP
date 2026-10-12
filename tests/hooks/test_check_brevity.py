"""Tests for the brevity budget check.

@see [[scripts/check_brevity.py]]
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

CONFIG = """
[tool.agent-kit.brevity]
bullet-files = ["docs/release_notes.rst"]
skip-dirs = ["generated"]
skip-paths = ["docs/handbook"]
budget = [
  { patterns = ["notes/*.md"], words = 20, sentences = true },
  { patterns = ["*/AGENTS.md"], lines = 60 },
]

[tool.agent-kit.brevity.existing]
"docs/old.md|words" = 30
"""


@pytest.fixture(scope="module")
def brevity() -> ModuleType:
    """Load scripts/check_brevity.py."""
    spec = importlib.util.spec_from_file_location(
        "check_brevity", ROOT / "scripts" / "check_brevity.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A repository with project budgets, a cap, and skips in pyproject.toml."""
    write(tmp_path, "pyproject.toml", CONFIG)
    return tmp_path


def run(
    brevity: ModuleType, root: Path, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    code = brevity.main([*args, "--root", str(root)])
    return code, capsys.readouterr().out


def test_words_skip_frontmatter_code_and_comments(brevity: ModuleType) -> None:
    text = "---\ntitle: a b c\n---\none two\n```\nx y z\n```\n<!-- note -->\nthree\n"
    assert brevity.words(brevity.prose(text, ".md")) == 3
    rst = "Intro words::\n\n    code here\n    more\n\nafter\n"
    assert brevity.words(brevity.prose(rst, ".rst")) == 3


def test_rst_section_underlines_are_not_fences(brevity: ModuleType) -> None:
    """RST ``~~~`` underlines keep the next section measured."""
    rst = "Fixed\n~~~~~\n\n- one two\n\nAdded\n~~~~~\n\n- three four\n"
    assert brevity.words(brevity.prose(rst, ".rst")) == 6


def test_long_sentences_counted_per_unit(brevity: ModuleType) -> None:
    long = " ".join(["word"] * 36) + "."
    text = f"{long} Short one.\n\n- {long}\n- item\n| a | {long} |\n"
    assert brevity.long_sentences(text) == 3


def test_over_budget_skill_fails(
    brevity: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(
        tmp_path, ".agents/skills/demo/SKILL.md", "---\nname: demo\n---\n" + "w " * 501
    )
    code, out = run(brevity, tmp_path, capsys)
    assert code == 1
    assert ".agents/skills/demo/SKILL.md: 501 > 500 (words)" in out


def test_vendored_files_exempt(
    brevity: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = "| Skill | Commit |\n|---|---|\n| demo, other (merged) | abc |\n"
    write(tmp_path, ".agents/skills/VENDORED.md", table)
    write(tmp_path, ".agents/skills/demo/SKILL.md", "w " * 900)
    write(tmp_path, ".claude/agents/VENDORED.md", table.replace("demo", "helper"))
    write(tmp_path, ".claude/agents/helper.md", "line\n" * 90)
    assert run(brevity, tmp_path, capsys) == (0, "")


def test_release_bullet_and_notebook_cells(
    brevity: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tmp_path = project
    bullet = "- " + "w " * 40 + "\n  " + "w " * 21 + "\n- short\n"
    write(tmp_path, "docs/release_notes.rst", "Notes\n=====\n\n" + bullet)
    cells = [
        {"cell_type": "markdown", "source": ["w " * 151]},
        {"cell_type": "code", "source": ["x = 1\n"] * 41},
    ]
    write(tmp_path, "examples/demo.ipynb", json.dumps({"cells": cells}))
    code, out = run(brevity, tmp_path, capsys)
    assert code == 1
    assert "docs/release_notes.rst:4: 61 > 60 (release-note bullet words)" in out
    assert "examples/demo.ipynb[cell 0]: 151 > 150 (markdown cell words)" in out
    assert "examples/demo.ipynb[cell 1]: 41 > 40 (code cell lines)" in out


def test_docstring_and_nested_agents_lines(
    brevity: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "scripts/tool.py", '"""Doc.\n' + "\n" * 11 + '"""\n')
    write(tmp_path, "src/AGENTS.md", "rule\n" * 16)
    code, out = run(brevity, tmp_path, capsys)
    assert code == 1
    assert "scripts/tool.py: 13 > 12 (module docstring lines)" in out
    assert "src/AGENTS.md: 16 > 15 (lines)" in out


def test_list_prints_passing_files(
    brevity: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path, "README.md", "Short readme.\n")
    code, out = run(brevity, tmp_path, capsys, "--list")
    assert code == 0
    assert "README.md: 2 <= 900 (words)" in out


def test_project_budgets_win_over_the_shared_ones(
    brevity: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A project budget is matched first: nested AGENTS.md may have 60 lines here."""
    write(project, "src/AGENTS.md", "rule\n" * 60)
    write(project, "notes/a.md", "word " * 21)
    code, out = run(brevity, project, capsys)
    assert code == 1
    assert "src/AGENTS.md" not in out
    assert "notes/a.md: 21 > 20 (words)" in out


def test_existing_cap_allows_shrinking_not_growing(
    brevity: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A grandfathered file is capped at its recorded size, not at the budget."""
    write(project, "docs/old.md", "word " * 30)
    write(project, "notes/b.md", "word " * 10)
    assert run(brevity, project, capsys)[0] == 0
    cap = brevity.grandfathered(
        brevity.Measure("docs/old.md", 31, 0, "words"), {("docs/old.md", "words"): 30}
    )
    assert cap.over
    assert cap.kind == "words, existing size"


def test_skipped_folders_and_paths_are_not_measured(
    brevity: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(project, "generated/AGENTS.md", "rule\n" * 200)
    write(project, "a/generated/AGENTS.md", "rule\n" * 200)
    write(project, "docs/handbook/README.md", "word " * 2000)
    assert run(brevity, project, capsys, "--list") == (0, "")


def test_display_math_is_not_prose(brevity: ModuleType) -> None:
    text = "$$" + " x +" * 50 + "$$\n\nShort sentence.\n"
    assert brevity.long_sentences(brevity.prose(text, ".md")) == 0


def test_notebooks_are_measured_only_in_the_notebook_folders(
    brevity: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cell = json.dumps({"cells": [{"cell_type": "markdown", "source": ["w " * 200]}]})
    write(tmp_path, "other/demo.ipynb", cell)
    assert run(brevity, tmp_path, capsys) == (0, "")
    write(
        tmp_path,
        "pyproject.toml",
        '[tool.agent-kit.brevity]\nnotebook-dirs = ["other"]\n',
    )
    assert run(brevity, tmp_path, capsys)[0] == 1
