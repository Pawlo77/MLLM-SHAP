"""Fixtures that load repository scripts as modules and build temporary vaults."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from _vault_helpers import GLOSSARY, MODULE_BODY, PYPROJECT, note, write

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` (or ``scripts/hooks/<name>.py``) by path."""
    path = SCRIPTS / f"{name}.py"
    if not path.is_file():
        path = SCRIPTS / "hooks" / f"{name}.py"
    for folder in (SCRIPTS, SCRIPTS / "hooks"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def script() -> Callable[[str], ModuleType]:
    """Return a loader for repository scripts."""
    return load_script


@pytest.fixture
def clean_vault(tmp_path: Path) -> Path:
    """Build a minimal vault that passes every check."""
    root = tmp_path / "vault"
    write(root, "scripts/glossary.yaml", GLOSSARY)
    write(root, "pyproject.toml", PYPROJECT)
    write(
        root,
        "src/mllm_shap/__init__.py",
        '"""Demo package.\n\n@domain core\n@see [[knowledge/architecture/core]]\n"""\n',
    )
    write(
        root,
        "src/mllm_shap/estimation/__init__.py",
        '"""Estimation.\n\n@domain estimation\n'
        '@see [[knowledge/architecture/estimation]]\n"""\n',
    )
    write(
        root,
        "src/mllm_shap/estimation/exact.py",
        '"""Exact estimator.\n\n@domain estimation\n'
        "@see [[knowledge/architecture/estimation]]\n"
        '@tested-by [[tests/demo/test_exact.py]]\n"""\n\n\n'
        'class ExactEstimator:\n    """Enumerate coalitions."""\n',
    )
    write(
        root,
        "tests/demo/test_exact.py",
        '"""Tests.\n\n@see [[src/mllm_shap/estimation/exact.py]]\n"""\n',
    )
    write(
        root,
        "knowledge/Map.md",
        note(
            "Map",
            "map",
            "[[core]] · [[estimation]] · [[Coalition]] · "
            "[[C01 - Demo issue]] · "
            "[[Demo decisions#2026-10-07 — Demo decision|Demo decision]] · "
            "[[Papers#Shapley 1953 — Demo|Shapley 1953]] · [[plans/2026-10-07-demo]]\n",
        ),
    )
    related = 'related-code:\n  - "[[src/mllm_shap/estimation/exact.py]]"\n'
    write(
        root,
        "knowledge/architecture/estimation.md",
        note("estimation", "module", MODULE_BODY, extra=related),
    )
    write(
        root,
        "knowledge/architecture/core.md",
        note("core", "module", MODULE_BODY.replace("Exact enumeration", "Exports")),
    )
    write(
        root,
        "knowledge/concepts/Coalition.md",
        note(
            "Coalition",
            "concept",
            "A coalition is a subset of players; complementary contribution (CC) "
            "pairs it with its complement. Back to [[Map]].\n",
        ),
    )
    write(
        root,
        "knowledge/issues/C01 - Demo issue.md",
        note(
            "C01 - Demo issue",
            "issue",
            "## Symptom\n\n- x\n\n## Cause\n\n- y\n\n## Repro\n\n- z\n\n"
            "## Fix\n\n- w\n",
            extra="status: open\nseverity: major\n",
        ),
    )
    write(
        root,
        "knowledge/decisions/Demo decisions.md",
        note(
            "Demo decisions",
            "decision-log",
            "## 2026-10-08 — Later decision\n\nDecision: b.\n\nWhy: w.\n\n"
            "Consequences: c.\n\n"
            "## 2026-10-07 — Demo decision\n\nDecision: a.\n\nWhy: w.\n\n"
            "Consequences: c.\n",
        ),
    )
    write(
        root,
        "knowledge/papers/Papers.md",
        note(
            "Papers",
            "paper-list",
            "## Shapley 1953 — Demo\n\nCitation: c.\n\nClaim: p.\n\n"
            "Used by: [[Map]].\n",
        ),
    )
    write(
        root,
        "knowledge/generated/Open issues.md",
        note("Open issues", "generated", "[[C01 - Demo issue]]\n", created=False),
    )
    write(
        root,
        "knowledge/Dashboard.md",
        note("Dashboard", "dashboard", "![[Open issues]] [[Map]]\n"),
    )
    write(
        root,
        "plans/2026-10-07-demo.md",
        note(
            "Demo plan",
            "plan",
            "**Goal:** g\n\n**Architecture:** a\n\n"
            "**Tech Stack:** t\n\n[[Dashboard]]\n",
            extra="status: draft\n",
        ),
    )
    write(root, ".remember/remember.md", "# Handoff\n\nSee [[Map]].\n")
    return root
