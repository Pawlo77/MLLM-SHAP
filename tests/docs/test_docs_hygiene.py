"""Static documentation checks: phrases, links, toctrees and API coverage.

@see [[docs/scripts/check_docs.py]]
"""

from pathlib import Path

import check_docs


def test_docs_have_no_findings() -> None:
    """The documentation sources pass every static check."""
    findings = [str(finding) for finding in check_docs.run_checks()]
    assert not findings, "\n".join(findings)


def test_phrase_check_flags_stale_wording(tmp_path: Path) -> None:
    """Stale coalition-table wording and skills paths are reported."""
    page = tmp_path / "page.rst"
    page.write_text(
        "We evaluate 2^n - 1 coalitions.\nSee .cursor/skills/x.\n", encoding="utf-8"
    )
    messages = [finding.message for finding in check_docs.check_phrases([page])]
    assert any("stale phrase" in message for message in messages)
    assert "use .agents/skills" in messages


def test_doc_link_check_flags_missing_targets(tmp_path: Path) -> None:
    """A :doc: role pointing at a missing page is reported."""
    (tmp_path / "present.rst").write_text("Present\n", encoding="utf-8")
    page = tmp_path / "page.rst"
    page.write_text(":doc:`present` and :doc:`Label <absent>`\n", encoding="utf-8")
    findings = check_docs.check_doc_links([page], tmp_path)
    assert [finding.message for finding in findings] == [
        ":doc: target 'absent' missing"
    ]


def test_doc_link_check_resolves_copied_notebooks(tmp_path: Path) -> None:
    """``notebooks/<name>`` resolves to ``examples/<name>.ipynb`` before the copy."""
    docs, examples = tmp_path / "docs", tmp_path / "examples"
    docs.mkdir()
    examples.mkdir()
    (examples / "demo.ipynb").write_text("{}", encoding="utf-8")
    page = docs / "page.rst"
    page.write_text(":doc:`notebooks/demo` :doc:`notebooks/gone`\n", encoding="utf-8")
    findings = check_docs.check_doc_links([page], docs, examples)
    assert [finding.message for finding in findings] == [
        ":doc: target 'notebooks/gone' missing"
    ]


def test_toctree_entries_are_parsed(tmp_path: Path) -> None:
    """Toctree entries are read and options are skipped."""
    index = tmp_path / "index.rst"
    index.write_text(
        ".. toctree::\n   :hidden:\n\n   guides/a\n   theory/b\n\nText\n",
        encoding="utf-8",
    )
    assert check_docs.toctree_entries(index) == {"guides/a", "theory/b"}


def test_repo_link_check_flags_missing_files(tmp_path: Path) -> None:
    """Links to files of this repository on ``main`` must exist locally."""
    page = tmp_path / "page.rst"
    page.write_text(
        "`ok <https://github.com/Pawlo77/nlp-shap/blob/main/README.md>`_\n"
        "`gone <https://github.com/Pawlo77/nlp-shap/blob/main/nope.md>`_\n",
        encoding="utf-8",
    )
    findings = check_docs.check_repo_links([page])
    assert [finding.message for finding in findings] == [
        "repository file 'nope.md' missing"
    ]
