"""Template drift: files edited after the template rendered them are reported.

@see [[scripts/template_drift.py]]
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

ANSWERS = "_commit: v1.0.0\n_src_path: gh:o/t\nowner: me\n"


@pytest.fixture(scope="module")
def drift() -> ModuleType:
    """Load scripts/template_drift.py."""
    spec = importlib.util.spec_from_file_location(
        "template_drift", ROOT / "scripts" / "template_drift.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_answers(drift: ModuleType) -> None:
    assert drift.parse_answers(ANSWERS) == {
        "_commit": "v1.0.0",
        "_src_path": "gh:o/t",
        "owner": "me",
    }


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, text in {".copier-answers.yml": ANSWERS, **files}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _render(files: dict[str, str]):
    def render(answers: dict[str, str], dest: Path) -> None:
        assert answers["_commit"] == "v1.0.0"
        for rel, text in {".copier-answers.yml": "different\n", **files}.items():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            (dest / rel).write_text(text, encoding="utf-8")

    return render


def test_clean_repository_has_no_drift(drift: ModuleType, tmp_path: Path) -> None:
    root = _repo(tmp_path, {"a.txt": "x\n", "sub/b.txt": "y\n", "own.txt": "mine\n"})
    render = _render({"a.txt": "x\n", "sub/b.txt": "y\n"})
    assert drift.main(["--root", str(root)], render) == 0


def test_edited_and_missing_files_are_reported(
    drift: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _repo(tmp_path, {"a.txt": "edited\n"})
    render = _render({"a.txt": "x\n", "gone.txt": "z\n"})
    assert drift.main(["--root", str(root)], render) == 1
    out = capsys.readouterr().out
    assert "drift: a.txt\n" in out
    assert "drift: gone.txt (missing)" in out


def test_not_from_the_template(drift: ModuleType, tmp_path: Path) -> None:
    assert drift.main(["--root", str(tmp_path)]) == 2
