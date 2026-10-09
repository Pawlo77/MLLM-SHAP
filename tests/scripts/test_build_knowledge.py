"""Tests for the generated-notes builder and graph report.

@see [[scripts/build_knowledge.py]]
"""

import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def builder(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/build_knowledge.py."""
    return script("build_knowledge")


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """Copy the files the builder reads into a temporary root."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "src" / "nlp_shap", root / "src" / "nlp_shap")
    shutil.copytree(
        REPO / "knowledge",
        root / "knowledge",
        ignore=shutil.ignore_patterns("generated"),
    )
    for rel in (
        "pyproject.toml",
        "docs/release_notes.rst",
        "tests/benchmarks/baselines.json",
        "tests/benchmarks/baseline.json",
        "scripts/glossary.yaml",
    ):
        source = REPO / rel
        if source.is_file():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, root / rel)
    return root


def test_build_is_idempotent_and_check_detects_drift(
    builder: ModuleType, repo_copy: Path
) -> None:
    assert builder.main(["--check"], root=repo_copy) == 1
    assert builder.main([], root=repo_copy) == 0
    generated = repo_copy / builder.GENERATED
    first = {path.name: path.read_text() for path in generated.iterdir()}
    assert builder.main([], root=repo_copy) == 0
    second = {path.name: path.read_text() for path in generated.iterdir()}
    assert first == second
    assert builder.main(["--check"], root=repo_copy) == 0
    issue = repo_copy / "knowledge" / "issues" / "Fixed issues.md"
    text = issue.read_text()
    flipped = (
        text.replace("status: fixed", "status: open", 1)
        if "status: fixed" in text
        else text.replace("status: open", "status: fixed", 1)
    )
    issue.write_text(flipped)
    assert builder.main(["--check"], root=repo_copy) == 1


def test_benchmark_rows_read_pytest_benchmark_medians(builder: ModuleType) -> None:
    data = {
        "machine_info": {"cpu": {"count": 14}},
        "benchmarks": [
            {"name": "test_b", "stats": {"median": 0.5, "mean": 0.6}},
            {"name": "test_a", "stats": {"median": 0.0012, "mean": 0.002}},
        ],
    }
    table_header, rows = builder.benchmark_rows(data)
    assert table_header.startswith("| Benchmark | Median (ms) |")
    assert rows == ["| `test_a` | 1.200 |", "| `test_b` | 500.000 |"]


def test_benchmark_rows_flatten_legacy_baselines(builder: ModuleType) -> None:
    table_header, rows = builder.benchmark_rows({"medians_s": {"scheduler": 0.25}})
    assert table_header.startswith("| Benchmark | Value |")
    assert rows == ["| `medians_s.scheduler` | 0.25 |"]


def test_generated_notes_are_marked(builder: ModuleType, repo_copy: Path) -> None:
    for path, content in builder.build_all(repo_copy).items():
        assert content.startswith("---\n"), path
        assert "generated: true" in content.split("---")[1]
        assert "Do not edit by hand" in content


def test_open_issue_lines_sorted_by_severity(
    builder: ModuleType, repo_copy: Path
) -> None:
    lines = builder.open_issue_lines(repo_copy)
    assert lines
    ranks = []
    for line in lines:
        severity, status = line.split("[", 1)[1].split("]")[0].split(", ")
        ranks.append((status == "deferred", builder.SEVERITY_ORDER[severity]))
    assert ranks == sorted(ranks)


def test_release_sections_counts_entries(builder: ModuleType) -> None:
    text = (
        "Release notes\n=============\n\nUnreleased\n----------\n\n"
        "0.2.0 (2026-01-02)\n------------------\n\nAdded\n~~~~~\n\n- a\n- b\n\n"
        "Fixed\n~~~~~\n\n- c\n\nDocumentation\n~~~~~~~~~~~~~\n\n- d\n"
    )
    sections = builder.release_sections(text)
    assert sections[0] == ("Unreleased", "", {})
    assert sections[1] == (
        "0.2.0",
        "2026-01-02",
        {"Added": 2, "Fixed": 1, "Documentation": 1},
    )


def test_config_keys_cover_nested_models(builder: ModuleType) -> None:
    content = builder.build_config_keys(REPO)
    assert "`explanation.budget.fraction`" in content
    assert "`backend.kind`" in content


def test_graph_neighbors_lists_imports(
    builder: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    status = builder.main(
        ["--graph", "--neighbors", "src/nlp_shap/estimation/exact.py"], root=REPO
    )
    assert status == 0
    output = capsys.readouterr().out
    assert "src/nlp_shap/domain/coalition.py" in output


def test_decisions_table_lists_log_sections(
    builder: ModuleType, clean_vault: Path
) -> None:
    log = clean_vault / "knowledge" / "decisions" / "Demo decisions.md"
    log.write_text(
        log.read_text().replace(
            "2026-10-07 — Demo decision",
            "2026-10-07 — Demo decision (superseded 2026-10-08)",
        )
    )
    rows = [
        line
        for line in builder.build_decisions(clean_vault).splitlines()
        if line.startswith("| 2026")
    ]
    assert rows == [
        (
            "| 2026-10-08 | demo | [[Demo decisions#2026-10-08 — Later decision"
            "\\|Later decision]] | accepted |"
        ),
        (
            "| 2026-10-07 | demo | [[Demo decisions#2026-10-07 — Demo decision "
            "(superseded 2026-10-08)\\|Demo decision]] | superseded 2026-10-08 |"
        ),
    ]
