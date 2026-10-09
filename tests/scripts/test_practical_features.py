"""Tests for the changed-path to practical-feature mapping.

@see [[scripts/practical_features.py]]
"""

import fnmatch
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from tests.practical._harness import SCENARIOS

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "tests" / "practical" / "feature_map.yaml"


@pytest.fixture(scope="module")
def features(script: Callable[[str], ModuleType]) -> ModuleType:
    """Load scripts/practical_features.py."""
    return script("practical_features")


def test_every_src_module_is_mapped(features: ModuleType) -> None:
    """Each module under src/ matches a glob that names ids or ``none``."""
    feature_map = features.load_map(MAP)
    modules = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob("src/**/*.py"))
    unmapped = [
        m for m in modules if not any(fnmatch.fnmatch(m, g) for g in feature_map.paths)
    ]
    assert not unmapped, f"add these to feature_map.yaml: {unmapped}"


def test_mapped_ids_name_scenarios(features: ModuleType) -> None:
    """Every id or glob in the map and the smoke set matches a scenario."""
    feature_map = features.load_map(MAP)
    ids = {i for value in feature_map.paths.values() for i in value}
    unknown = [
        i for i in ids | set(feature_map.smoke) if not fnmatch.filter(SCENARIOS, i)
    ]
    assert not unknown


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write(repo: Path, rel: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'"""{rel}."""\n', encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with a ``main`` commit and a checked-out feature branch."""
    _git(tmp_path, "init", "-q", "-b", "main")
    for rel in ("README.md", "src/nlp_shap/cli.py", "src/nlp_shap/_logging.py"):
        _write(tmp_path, rel)
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "chore: base")
    _git(tmp_path, "switch", "-qc", "feat/x")
    return tmp_path


def test_committed_staged_and_untracked_changes_map(
    features: ModuleType, repo: Path
) -> None:
    """Committed, modified and new files count; unmapped and ``none`` add nothing."""
    _write(repo, "src/nlp_shap/estimation/neyman.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feat: neyman")
    (repo / "src/nlp_shap/cli.py").write_text("changed\n", encoding="utf-8")
    _write(repo, "src/nlp_shap/viz/render.py")
    (repo / "src/nlp_shap/_logging.py").write_text("changed\n", encoding="utf-8")

    ids = features.feature_ids(
        features.changed_paths("main", repo), features.load_map(MAP)
    )

    assert ids == ["cli.explain", "cli.reanalyze", "estimator.neyman_cc", "viz.render"]


def test_orchestrator_change_runs_the_smoke_set(
    features: ModuleType, repo: Path
) -> None:
    """Runner or orchestrator changes map to the smoke scenarios."""
    feature_map = features.load_map(MAP)
    _write(repo, "src/nlp_shap/pipeline/orchestrator.py")

    ids = features.feature_ids(features.changed_paths("main", repo), feature_map)

    assert ids == sorted(feature_map.smoke)
    assert len(feature_map.smoke) == 3


def test_docs_only_change_prints_an_empty_line(
    features: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A branch without mapped src/ changes prints an empty line."""
    (repo / "README.md").write_text("more\n", encoding="utf-8")
    assert features.main(["--base", "main", "--root", str(repo)]) == 0
    assert capsys.readouterr().out == "\n"
