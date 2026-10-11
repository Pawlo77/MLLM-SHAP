"""Tests for the vault consistency checker.

@see [[scripts/check_vault.py]]
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from _vault_helpers import note, write


@pytest.fixture(scope="module")
def checker(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/check_vault.py."""
    return script("check_vault")


def _errors(checker: ModuleType, root: Path, **flags: bool) -> list[str]:
    findings, _, _ = checker.run(root, **flags)
    return [str(item) for item in findings if not item.warning]


def _warnings(checker: ModuleType, root: Path) -> list[str]:
    findings, _, _ = checker.run(root)
    return [str(item) for item in findings if item.warning]


def test_clean_vault_passes(checker: ModuleType, clean_vault: Path) -> None:
    assert _errors(checker, clean_vault) == []
    assert _warnings(checker, clean_vault) == []


def _replace(rel: str, old: str, new: str) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        assert old in text
        path.write_text(text.replace(old, new), encoding="utf-8")

    return mutate


def _add(rel: str, text: str) -> Callable[[Path], None]:
    def mutate(root: Path) -> None:
        write(root, rel, text)

    return mutate


ESTIMATION = "knowledge/architecture/estimation.md"
CONCEPT = "knowledge/concepts/Coalition.md"
DECISIONS = "knowledge/decisions/Demo decisions.md"
PAPERS = "knowledge/papers/Papers.md"

VIOLATIONS = [
    (
        "invalid yaml",
        _replace(CONCEPT, "domain: demo", "domain: [demo"),
        "not valid YAML",
    ),
    ("missing field", _replace(CONCEPT, "domain: demo\n", ""), "has no 'domain'"),
    ("bad type", _replace(CONCEPT, "type: concept", "type: essay"), "type 'essay'"),
    (
        "bad status",
        _replace(
            "knowledge/issues/C01 - Demo issue.md", "status: open", "status: solved"
        ),
        "status 'solved'",
    ),
    (
        "missing section",
        _replace(ESTIMATION, "## Invariants", "## Rules"),
        "missing section '## Invariants'",
    ),
    ("broken link", _replace(CONCEPT, "[[Map]]", "[[Nowhere]]"), "broken link"),
    (
        "related code missing",
        _replace(ESTIMATION, 'estimation/exact.py]]"', 'estimation/gone.py]]"'),
        "related-code target missing",
    ),
    (
        "isolated note",
        _add("knowledge/concepts/Lonely.md", note("Lonely", "concept", "Alone.\n")),
        "isolated note",
    ),
    (
        "abbreviation",
        _replace(CONCEPT, "Back to", "MC sampling. Back to"),
        "abbreviation MC",
    ),
    (
        "secret",
        _replace(CONCEPT, "Back to", "ghp_" + "a" * 36 + " Back to"),
        "GitHub personal access token",
    ),
    (
        "source pointer",
        _replace(PAPERS, "Claim: p.", "Claim: see p. 12."),
        "source pointer",
    ),
    (
        "broken heading link",
        _replace("knowledge/Map.md", "— Demo decision|", "— Missing|"),
        "broken heading link [[Demo decisions#2026-10-07 — Missing]]",
    ),
    (
        "entry field missing",
        _replace(
            DECISIONS, "Why: w.\n\nConsequences: c.\n\n## 2026-10-07", "## 2026-10-07"
        ),
        "entry '2026-10-08 — Later decision' has no 'Why:' line",
    ),
    (
        "entry heading malformed",
        _replace(PAPERS, "## Shapley 1953 — Demo", "## Shapley - Demo"),
        "entry heading 'Shapley - Demo' malformed",
    ),
    (
        "entries out of order",
        _replace(DECISIONS, "2026-10-08 — Later", "2026-10-06 — Later"),
        "entries are not listed newest first",
    ),
    (
        "concept paragraph",
        _replace(CONCEPT, "A coalition is", "## Definition\n\nA coalition is"),
        "first paragraph",
    ),
    (
        "code link",
        _replace(
            "src/mllm_shap/estimation/exact.py",
            '"""Enumerate coalitions."""',
            '"""Enumerate coalitions, see [[Missing note]]."""',
        ),
        "broken link [[Missing note]]",
    ),
    (
        "plan header",
        _replace("plans/2026-10-07-demo.md", "**Tech Stack:** t", "no stack"),
        "plan header line '**Tech Stack:**' missing",
    ),
    (
        "duplicate name",
        _add(
            "knowledge/handbook/Coalition.md",
            note("Coalition", "handbook", "[[Map]]\n"),
        ),
        "note name 'Coalition' also used",
    ),
    (
        "stray templater file",
        _add("src/mllm_shap/<% tp.file.title %>/__init__.py.md", ""),
        "stray file created by Obsidian",
    ),
    (
        "stray empty note",
        _add("knowledge/concepts/Untitled.md", ""),
        "stray file created by Obsidian",
    ),
    (
        "skill link",
        _add(".agents/skills/demo/SKILL.md", "See [[No such note]].\n"),
        "broken link [[No such note]]",
    ),
    (
        "template placeholder link",
        _add("knowledge/_templates/demo.md", "Package: [[<package note>]].\n"),
        "broken link [[<package note>]]",
    ),
    (
        "markdown link",
        _replace(CONCEPT, "A coalition is", "See [licence](LICENSE). A coalition is"),
        "broken markdown link (LICENSE)",
    ),
    (
        "dot-folder link",
        _replace(
            CONCEPT, "A coalition is", "See [[.remember/remember]]. A coalition is"
        ),
        "broken link [[.remember/remember]]",
    ),
]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [pytest.param(mutate, expected, id=name) for name, mutate, expected in VIOLATIONS],
)
def test_each_rule_reports_one_violation(
    checker: ModuleType,
    clean_vault: Path,
    mutate: Callable[[Path], None],
    expected: str,
) -> None:
    mutate(clean_vault)
    errors = _errors(checker, clean_vault)
    assert any(expected in error for error in errors), errors


def test_missing_tags_warn_until_strict(checker: ModuleType, clean_vault: Path) -> None:
    _replace("src/mllm_shap/estimation/exact.py", "@domain estimation\n", "")(
        clean_vault
    )
    assert _errors(checker, clean_vault) == []
    assert any("missing @domain" in item for item in _warnings(checker, clean_vault))
    strict = _errors(checker, clean_vault, strict_tags=True)
    assert any("missing @domain" in item for item in strict)


def test_test_file_needs_see_tag(checker: ModuleType, clean_vault: Path) -> None:
    write(clean_vault, "tests/demo/test_other.py", '"""No tag."""\n')
    assert any("missing @see" in item for item in _warnings(checker, clean_vault))


def test_plan_without_frontmatter_warns_until_strict(
    checker: ModuleType, clean_vault: Path
) -> None:
    write(clean_vault, "plans/2026-10-07-report.md", "# Report\n\n[[Map]]\n")
    assert _errors(checker, clean_vault) == []
    assert any("no frontmatter" in item for item in _warnings(checker, clean_vault))
    assert any(
        "no frontmatter" in item for item in _errors(checker, clean_vault, strict=True)
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_stale_architecture_note_warns_then_fails_strict(
    checker: ModuleType, clean_vault: Path
) -> None:
    _replace(ESTIMATION, "created: 2026-10-07\n", "created: 2000-01-01\n")(clean_vault)
    _git(clean_vault, "init", "-q", "-b", "main")
    _git(clean_vault, "add", ".")
    _git(clean_vault, "commit", "-q", "-m", "base")
    _git(clean_vault, "update-ref", "refs/remotes/origin/main", "HEAD")
    exact = clean_vault / "src/mllm_shap/estimation/exact.py"
    exact.write_text(exact.read_text(encoding="utf-8") + "# change\n")
    _git(clean_vault, "commit", "-q", "-am", "change")
    warnings = _warnings(checker, clean_vault)
    assert any("older than package change" in item for item in warnings)
    strict = _errors(checker, clean_vault, strict=True)
    assert any("knowledge/architecture/estimation.md" in item for item in strict)


def test_main_quiet_prints_only_errors(
    checker: ModuleType, clean_vault: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert checker.main(["--quiet", "--root", str(clean_vault)]) == 0
    assert capsys.readouterr().out == ""
    _replace(CONCEPT, "[[Map]]", "[[Nowhere]]")(clean_vault)
    assert checker.main(["--quiet", "--root", str(clean_vault)]) == 1
    output = capsys.readouterr().out
    assert "knowledge/concepts/Coalition.md:" in output
    assert "broken link [[Nowhere]]" in output


@pytest.mark.parametrize(
    "rel",
    ["src/pkg/AGENTS.md", "plans/notes/extra.md", ".claude/rules/rule.md", "x/y/z.md"],
)
def test_broken_links_are_found_in_every_markdown_file(
    checker: ModuleType, clean_vault: Path, rel: str
) -> None:
    """Obsidian creates a stray note from any broken link, so every file is checked."""
    write(clean_vault, rel, "See [[Missing note]].\n")
    assert any(
        rel in error and "broken link [[Missing note]]" in error
        for error in _errors(checker, clean_vault)
    )


def test_ignored_markdown_is_not_link_checked(
    checker: ModuleType, clean_vault: Path
) -> None:
    write(clean_vault, ".venv/lib/site.md", "See [[Missing note]].\n")
    write(clean_vault, "docs/_build/page.md", "See [[Missing note]].\n")
    assert _errors(checker, clean_vault) == []
