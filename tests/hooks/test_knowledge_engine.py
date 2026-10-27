"""The shared knowledge builder describes a package from its source and pyproject.

@see [[scripts/build_knowledge.py]]
"""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.hooks._hookrun import ROOT

PYPROJECT = """\
[project]
name = "demo"
requires-python = ">=3.12"
dependencies = ["requests>=2"]

[project.optional-dependencies]
viz = ["matplotlib>=3"]

[project.scripts]
demo = "demo.cli:main"

[dependency-groups]
dev = ["pytest"]
"""


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    """Load scripts/build_knowledge.py."""
    spec = importlib.util.spec_from_file_location(
        "build_knowledge_under_test", ROOT / "scripts" / "build_knowledge.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def project(
    tmp_path: Path, builder: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """A package with a public class, a CLI and a vault folder, no project builders."""
    monkeypatch.setattr(builder, "load_extra", lambda: None)
    _write(tmp_path, "pyproject.toml", PYPROJECT)
    _write(
        tmp_path,
        "src/demo/__init__.py",
        '"""Demo package."""\n\nfrom .core import Thing\n\n__all__ = ["Thing"]\n',
    )
    _write(
        tmp_path,
        "src/demo/core.py",
        '"""Core types."""\n\n\nclass Thing:\n    """A thing that does work."""\n',
    )
    _write(
        tmp_path,
        "src/demo/cli.py",
        '"""Command line."""\n\n\ndef main():\n    """Run."""\n',
    )
    (tmp_path / "knowledge").mkdir()
    return tmp_path


def test_notes_describe_the_package(builder: ModuleType, project: Path) -> None:
    notes = {path.stem: text for path, text in builder.build_all(project).items()}
    assert set(notes) == {"Modules", "Public API", "Entry points", "Dependencies"}
    assert "| [[src/demo/core.py]] | Core types. |" in notes["Modules"]
    assert "Architecture note" not in notes["Modules"]  # no module notes to list
    assert (
        "| `Thing` | [[src/demo/core.py]] | A thing that does work. |"
        in notes["Public API"]
    )
    assert (
        "| console_scripts | `demo` | `demo.cli:main` | [[src/demo/cli.py]] |"
        in notes["Entry points"]
    )
    assert "| Extra | viz | matplotlib>=3 |" in notes["Dependencies"]
    assert "| Group | dev | pytest |" in notes["Dependencies"]


def test_module_notes_add_the_architecture_column(
    builder: ModuleType, project: Path
) -> None:
    _write(
        project,
        "knowledge/Core.md",
        '---\ntitle: Core\ntype: module\nrelated-code:\n  - "[[src/demo/core.py]]"\n'
        "---\n",
    )
    modules = next(
        text
        for path, text in builder.build_all(project).items()
        if path.stem == "Modules"
    )
    assert "| Architecture note |" in modules
    assert "| [[src/demo/core.py]] | Core types. | 5 | [[Core]] |" in modules
    assert "- [[src/demo/cli.py]]" in modules  # listed by no module note


def test_settings_choose_generators_and_frontmatter(
    builder: ModuleType, project: Path
) -> None:
    table = (
        '[tool.agent-kit.knowledge]\ngenerators = ["public-api"]\n'
        'parent = "[[Map]]"\nupdated = true\n'
    )
    _write(project, "pyproject.toml", PYPROJECT + table)
    ((path, text),) = builder.build_all(project).items()
    assert path.name == "Public API.md"
    assert 'parent: "[[Map]]"' in text
    assert "updated: " in text
    assert "@DATE@" not in text


def test_updated_date_is_kept_while_the_body_is_unchanged(
    builder: ModuleType, project: Path
) -> None:
    table = '[tool.agent-kit.knowledge]\ngenerators = ["public-api"]\nupdated = true\n'
    _write(project, "pyproject.toml", PYPROJECT + table)
    ((path, text),) = builder.build_all(project).items()
    path.parent.mkdir(parents=True)
    path.write_text(
        text.replace(text.split("updated: ")[1].split("\n")[0], "2020-01-01")
    )
    ((_, again),) = builder.build_all(project).items()
    assert "updated: 2020-01-01" in again
    _write(project, "src/demo/__init__.py", '"""Demo."""\n\n__all__ = []\n')
    ((_, changed),) = builder.build_all(project).items()
    assert "updated: 2020-01-01" not in changed


def test_empty_public_api_ends_in_one_newline(
    builder: ModuleType, project: Path
) -> None:
    _write(project, "src/demo/__init__.py", '"""Demo."""\n\n__all__ = []\n')
    built = builder.build_all(project)
    api = next(text for path, text in built.items() if path.name == "Public API.md")
    assert api.endswith("|---|---|---|\n")


def test_check_reports_stale_notes_and_write_fixes_them(
    builder: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert builder.main(["--check"], project) == 1
    assert "stale; run `make knowledge`" in capsys.readouterr().out
    assert builder.main([], project) == 0
    assert builder.main(["--check"], project) == 0


def test_project_without_a_vault_builds_nothing(
    builder: ModuleType, project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (project / "knowledge").rmdir()
    assert builder.main([], project) == 0
    assert "nothing to build" in capsys.readouterr().out
    assert not (project / "knowledge").exists()


def test_project_builders_and_main_override(
    builder: ModuleType, project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extra = SimpleNamespace(BUILDERS={"Extra": lambda root: "extra body\n"})
    monkeypatch.setattr(builder, "load_extra", lambda: extra)
    names = {path.stem for path in builder.build_all(project)}
    assert "Extra" in names
    extra.main = lambda argv, root: 7
    assert builder.main([], project) == 7
