"""Test selection for the pre-push hook: only tests a change can reach run.

@see [[scripts/select_tests.py]]
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT, git


@pytest.fixture(scope="module")
def selector() -> ModuleType:
    """Load scripts/select_tests.py."""
    spec = importlib.util.spec_from_file_location(
        "select_tests_under_test", ROOT / "scripts" / "select_tests.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PYPROJECT = (
    '[project]\nname = "pkg"\ndependencies = ["a"]\n\n'
    '[project.urls]\nHomepage = "https://example.com/old"\n\n[tool.agent-kit]\n'
)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A package with a shared module, two leaves and a subprocess-style test."""
    _write(
        tmp_path,
        {
            "pyproject.toml": PYPROJECT,
            "src/pkg/__init__.py": "from pkg import a, b\n",
            "src/pkg/core.py": "X = 1\n",
            "src/pkg/a.py": "from pkg.core import X\n",
            "src/pkg/b.py": "Y = 2\n",
            "src/pkg/cli.py": "Z = 3\n",
            "tests/helpers.py": "from pkg import a\n",
            "tests/test_a.py": "from pkg.a import X\n",
            "tests/test_b.py": "from pkg import b\n",
            "tests/test_cli.py": 'CMD = ["python", "-m", "pkg.cli"]\n',
            "tests/test_helper.py": "from tests.helpers import a\n",
            "tests/test_data.py": 'NAME = "settings.yaml"\n',
        },
    )
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "chore: base")
    git(tmp_path, "checkout", "-q", "-b", "work")
    return tmp_path


def _tests(selector: ModuleType, repo: Path, **files: str) -> list[str] | None:
    """Commit ``files`` (path with ``__`` for ``/``) on the branch; return the plan."""
    _write(repo, {k.replace("__", "/"): v for k, v in files.items()})
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "feat: change")
    return selector.plan(repo, "main")


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ("src__pkg__core.py", {"test_a.py", "test_helper.py"}),
        ("src__pkg__b.py", {"test_b.py"}),
        ("src__pkg__cli.py", {"test_cli.py"}),
        ("settings.yaml", {"test_data.py"}),
        ("tests__test_b.py", {"test_b.py"}),
    ],
)
def test_only_reachable_tests_are_selected(
    selector: ModuleType, repo: Path, changed: str, expected: set[str]
) -> None:
    """A leaf change selects its importers, string mentions and itself."""
    picked = _tests(selector, repo, **{changed: "X = 9\n"})
    assert picked is not None
    assert {Path(p).name for p in picked} == expected


@pytest.mark.parametrize("changed", ["docs__guide.md", "knowledge__note.md"])
def test_documentation_selects_nothing(
    selector: ModuleType, repo: Path, changed: str
) -> None:
    """Notes and docs are read by no test."""
    assert _tests(selector, repo, **{changed: "text\n"}) == []


@pytest.mark.parametrize("changed", ["tests__conftest.py", "uv.lock"])
def test_setup_changes_run_everything(
    selector: ModuleType, repo: Path, changed: str
) -> None:
    """Dependencies, build files and conftest move every test."""
    assert _tests(selector, repo, **{changed: "# x\n"}) is None


def test_deleted_python_file_runs_everything(selector: ModuleType, repo: Path) -> None:
    """A removed module cannot be traced, so the whole suite runs."""
    (repo / "src/pkg/b.py").unlink()
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "refactor: drop b")
    assert selector.plan(repo, "main") is None


def test_unknown_base_runs_everything(selector: ModuleType, repo: Path) -> None:
    """Without a merge base nothing can be told apart."""
    assert selector.plan(repo, "no-such-ref") is None


def test_a_dependency_change_runs_everything(selector: ModuleType, repo: Path) -> None:
    """Dependencies move every test."""
    changed = PYPROJECT.replace('["a"]', '["a", "b"]')
    assert _tests(selector, repo, **{"pyproject.toml": changed}) is None


@pytest.mark.parametrize(
    ("old", "new"),
    [("example.com/old", "example.com/new"), ("[tool.agent-kit]", "[tool.ruff]")],
)
def test_metadata_and_tool_settings_do_not_run_everything(
    selector: ModuleType, repo: Path, old: str, new: str
) -> None:
    """Project URLs and the settings of linters leave the tests alone."""
    picked = _tests(selector, repo, **{"pyproject.toml": PYPROJECT.replace(old, new)})
    assert picked == []


def test_an_uncommitted_change_is_read_whole(selector: ModuleType, repo: Path) -> None:
    """The first `git status` line starts with a space; its path stays whole."""
    (repo / "src/pkg/b.py").write_text("Y = 3\n", encoding="utf-8")
    picked = selector.plan(repo, "main")
    assert picked is not None
    assert {Path(p).name for p in picked} == {"test_b.py"}
