"""Tests for the brevity budget check.

@see [[scripts/check_brevity.py]]
"""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from _vault_helpers import write


@pytest.fixture(scope="module")
def brevity(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/check_brevity.py."""
    return script("check_brevity")


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
    brevity: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
