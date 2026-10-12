"""Tests for the upgrade checker and applier, with faked ``uv``, ``prek`` and ``gh``.

@see [[scripts/upgrade.py]]
"""

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

OLD_SHA, NEW_SHA, TAG_SHA = "a" * 40, "b" * 40, "c" * 40
PIN_OLD, PIN_NEW = "d" * 40, "e" * 40
PIN_LOCK = (
    'version = 1\n\n[[package]]\nname = "demo-lib"\nversion = "0.2.1"\n'
    'source = {{ git = "https://github.com/o/DEMO-LIB.git?branch=main#{sha}" }}\n'
)
LOCK_OUT = (
    "Resolved 3 packages in 1s\nUpdate numpy v2.5.1 -> v2.5.3\nAdd cloudpickle v3.1.2\n"
    "Would update pydantic v2.13.4 -> v2.14.0\n"
)
PREK_OUT = (
    "https://github.com/codespell-project/codespell\n"
    "  would update rev `v2.4.2` -> `v2.4.3`\n"
)
PIN_LOCK_OUT = f"Updated https://github.com/o/DEMO-LIB.git ({PIN_NEW})\n" + LOCK_OUT
LATEST = {
    "actions/checkout": "v7.1.0",
    "astral-sh/setup-uv": "v10.3.0",
    "github/codeql-action": "codeql-bundle-v2.27.2",
    "astral-sh/uv": "0.13.0",
    "blacksmithgu/obsidian-dataview": "0.5.70",
}
WORKFLOW = f"""jobs:
  test:
    steps:
      - uses: actions/checkout@v7.0.1
      - uses: astral-sh/setup-uv@{OLD_SHA} # v10.2.0
        with:
          version: "0.12.23"
      - uses: github/codeql-action/init@v4.38.2
      - uses: actions/checkout@v7.0.1
"""
PUBLISH = """jobs:
  pypi:
    steps:
      - uses: actions/checkout@v7.0.1
      - uses: astral-sh/setup-uv@v10.2.0
        with:
          version: "0.12.23"
"""


SHAP_LIKE = """
[tool.agent-kit.upgrade]
protected-workflows = ["publish.yaml"]
sync-args = "--all-groups --all-extras"
"""
PIN_LIKE = """
[tool.agent-kit.upgrade]
pin = "demo-lib"
pin-git = "DEMO-LIB"
"""


@pytest.fixture(scope="module")
def upgrade() -> ModuleType:
    """Load scripts/upgrade.py."""
    spec = importlib.util.spec_from_file_location(
        "upgrade", ROOT / "scripts" / "upgrade.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with every kind of pin, one version behind."""
    files = {
        "pyproject.toml": '[tool.uv]\nrequired-version = ">=0.12.23"\n' + SHAP_LIKE,
        "uv.lock": "version = 1\n",
        ".pre-commit-config.yaml": (
            "repos:\n  - repo: https://github.com/codespell-project/codespell\n"
            "    rev: v2.4.2\n"
        ),
        ".github/workflows/test.yaml": WORKFLOW,
        ".github/workflows/publish.yaml": PUBLISH,
        ".obsidian/community-plugins.json": '["dataview"]',
        ".obsidian/plugins/dataview/manifest.json": (
            '{\n  "id": "dataview",\n  "version": "0.5.68"\n}\n'
        ),
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


class FakeRunner:
    """Answers the commands ``scripts/upgrade.py`` runs; records every call."""

    def __init__(self, error: type[Exception]) -> None:
        self.calls: list[list[str]] = []
        self.error = error

    def __call__(self, argv: Sequence[str], cwd: Path) -> str:
        argv = list(argv)
        self.calls.append(argv)
        pinned = "demo-lib" in (cwd / "uv.lock").read_text(encoding="utf-8")
        if argv[:2] == ["uv", "lock"]:
            if "--dry-run" in argv:
                return PIN_LOCK_OUT if pinned else LOCK_OUT
            if pinned:
                moved = "--upgrade-package=demo-lib" in argv
                text = PIN_LOCK.format(sha=PIN_NEW if moved else PIN_OLD)
                (cwd / "uv.lock").write_text(text + "# upgraded\n", encoding="utf-8")
            else:
                (cwd / "uv.lock").write_text("version = 2\n", encoding="utf-8")
        elif argv[:3] == ["uv", "run", "prek"]:
            if "--dry-run" in argv:
                return PREK_OUT
            config = Path(argv[argv.index("-c") + 1])
            config.write_text(config.read_text().replace("v2.4.2", "v2.4.3"))
        elif argv[:2] == ["gh", "api"]:
            return self._gh(next(a for a in argv[2:] if a != "--paginate"))
        return ""

    def _gh(self, path: str) -> str:
        if path.endswith("/releases/latest"):
            return LATEST[path.removeprefix("repos/").removesuffix("/releases/latest")]
        if "/matching-refs/" in path:
            return "refs/tags/v4\nrefs/tags/v4.38.3\nrefs/tags/v4.38.2\n"
        if "/git/ref/tags/" in path:
            return f"tag {TAG_SHA}"
        if path.endswith(f"/git/tags/{TAG_SHA}"):
            return NEW_SHA
        raise self.error(path)


@pytest.fixture
def fake(upgrade: ModuleType) -> FakeRunner:
    """A runner that never touches the network."""
    return FakeRunner(upgrade.UpgradeError)


def test_check_lists_every_outdated_item(
    upgrade: ModuleType, repo: Path, fake: FakeRunner
) -> None:
    """One item per outdated pin; publish.yaml pins and uv are report-only."""
    found = {
        (i.kind, i.name, i.old, i.new, i.protected) for i in upgrade.check(fake, repo)
    }
    assert found == {
        ("python", "numpy", "2.5.1", "2.5.3", False),
        ("python", "pydantic", "2.13.4", "2.14.0", False),
        (
            "hook",
            "https://github.com/codespell-project/codespell",
            "v2.4.2",
            "v2.4.3",
            False,
        ),
        ("action", "actions/checkout", "v7.0.1", "v7.1.0", False),
        ("action", "astral-sh/setup-uv", "v10.2.0", "v10.3.0", False),
        ("action", "github/codeql-action/init", "v4.38.2", "v4.38.3", False),
        ("action", "actions/checkout", "v7.0.1", "v7.1.0", True),
        ("action", "astral-sh/setup-uv", "v10.2.0", "v10.3.0", True),
        ("uv", "uv", "0.12.23", "0.13.0", True),
        ("obsidian", "dataview", "0.5.68", "0.5.70", False),
    }
    latest_calls = [c for c in fake.calls if c[2].endswith("/releases/latest")]
    assert len(latest_calls) == len(LATEST)  # one lookup per repository


def test_apply_upgrades_all_but_protected(
    upgrade: ModuleType, repo: Path, fake: FakeRunner
) -> None:
    """Pins move to the latest tag (a sha pin to the tag's commit); publish stays."""
    changed, protected = upgrade.apply(fake, repo)
    workflow = (repo / ".github/workflows/test.yaml").read_text()
    assert "actions/checkout@v7.1.0\n" in workflow
    assert workflow.count("actions/checkout@v7.1.0") == 2
    assert f"astral-sh/setup-uv@{NEW_SHA} # v10.3.0" in workflow
    assert "github/codeql-action/init@v4.38.3" in workflow
    assert 'version: "0.12.23"' in workflow  # uv is protected by publish.yaml
    assert (repo / ".github/workflows/publish.yaml").read_text() == PUBLISH
    assert "rev: v2.4.3" in (repo / ".pre-commit-config.yaml").read_text()
    manifest = repo / ".obsidian/plugins/dataview/manifest.json"
    assert json.loads(manifest.read_text())["version"] == "0.5.70"
    assert sorted(changed) == [
        ".github/workflows/test.yaml",
        ".obsidian/plugins/dataview/manifest.json",
        ".pre-commit-config.yaml",
        "uv.lock",
    ]
    assert {(i.kind, i.where) for i in protected} == {
        ("action", ".github/workflows/publish.yaml"),
        ("uv", "pyproject.toml, .github/workflows"),
    }
    assert ["uv", "sync", "--all-groups", "--all-extras"] in fake.calls
    assert [*upgrade.INSTALL_PLUGINS, "--force"] in fake.calls


def test_apply_bumps_uv_together_without_a_publish_pin(
    upgrade: ModuleType, repo: Path, fake: FakeRunner
) -> None:
    """required-version and every workflow uv pin move together."""
    publish = repo / ".github/workflows/publish.yaml"
    publish.write_text(PUBLISH.replace('          version: "0.12.23"\n', ""))
    upgrade.apply(fake, repo)
    assert 'required-version = ">=0.13.0"' in (repo / "pyproject.toml").read_text()
    assert 'version: "0.13.0"' in (repo / ".github/workflows/test.yaml").read_text()


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [("v1.2.3", "v1.10.0", True), ("1.2", "1.2", False), ("v4", "bundle", False)],
)
def test_newer(upgrade: ModuleType, old: str, new: str, expected: bool) -> None:
    assert upgrade.newer(old, new) is expected


def test_main_exit_codes(
    upgrade: ModuleType,
    repo: Path,
    fake: FakeRunner,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--check exits 1 with items, --apply prints the changed files, errors exit 2."""
    monkeypatch.setattr(upgrade, "run", fake)
    assert upgrade.main(["--check", "--root", str(repo)]) == 1
    assert "(protected: report only)" in capsys.readouterr().out
    assert upgrade.main(["--apply", "--root", str(repo)]) == 0
    assert "changed  uv.lock" in capsys.readouterr().out

    def broken(argv: Sequence[str], cwd: Path) -> str:
        raise upgrade.UpgradeError("gh api failed")  # noqa: TRY003

    monkeypatch.setattr(upgrade, "run", broken)
    assert upgrade.main(["--check", "--root", str(repo)]) == 2
    assert "gh api failed" in capsys.readouterr().err


@pytest.fixture
def pin_repo(tmp_path: Path) -> Path:
    """A repository whose ``demo-lib`` commit is pinned; no protected workflow."""
    files = {
        "pyproject.toml": '[tool.uv]\nrequired-version = ">=0.12.23"\n' + PIN_LIKE,
        "uv.lock": PIN_LOCK.format(sha=PIN_OLD),
        ".pre-commit-config.yaml": (
            "repos:\n  - repo: https://github.com/codespell-project/codespell\n"
            "    rev: v2.4.2\n"
        ),
        ".github/workflows/test.yaml": WORKFLOW,
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def test_check_reports_the_pin_as_protected(
    upgrade: ModuleType, pin_repo: Path, fake: FakeRunner
) -> None:
    """The pinned commit shows up as report-only; uv is not protected here."""
    found = {
        (i.kind, i.name, i.old, i.new, i.protected)
        for i in upgrade.check(fake, pin_repo)
    }
    assert ("python", "demo-lib", PIN_OLD[:7], PIN_NEW[:7], True) in found
    assert ("python", "numpy", "2.5.1", "2.5.3", False) in found
    assert ("uv", "uv", "0.12.23", "0.13.0", False) in found


def test_apply_upgrades_all_but_the_pin(
    upgrade: ModuleType, pin_repo: Path, fake: FakeRunner
) -> None:
    """Packages, hooks, action pins and uv move; the pinned commit stays."""
    changed, protected = upgrade.apply(fake, pin_repo)
    assert PIN_OLD in (pin_repo / "uv.lock").read_text()
    assert 'version: "0.13.0"' in (pin_repo / ".github/workflows/test.yaml").read_text()
    assert sorted(changed) == [
        ".github/workflows/test.yaml",
        ".pre-commit-config.yaml",
        "pyproject.toml",
        "uv.lock",
    ]
    assert [i.name for i in protected] == ["demo-lib"]
    lock = ["uv", "lock", "--upgrade-package=numpy", "--upgrade-package=pydantic"]
    assert lock in fake.calls
    assert ["uv", "sync", "--all-groups"] in fake.calls


def test_move_pin_only_moves_the_pin(
    upgrade: ModuleType, pin_repo: Path, fake: FakeRunner
) -> None:
    """``--pin`` upgrades that one package and nothing else."""
    assert upgrade.move_pin(fake, pin_repo) == ["uv.lock"]
    assert PIN_NEW in (pin_repo / "uv.lock").read_text()
    assert ["uv", "lock", "--upgrade-package=demo-lib"] in fake.calls


def test_move_pin_needs_a_configured_pin(
    upgrade: ModuleType, repo: Path, fake: FakeRunner
) -> None:
    with pytest.raises(upgrade.UpgradeError, match="no pin"):
        upgrade.move_pin(fake, repo)


@pytest.fixture
def managed_repo(pin_repo: Path) -> Path:
    """A repository from the template: ``prek.yaml`` and the prek config are shipped."""
    (pin_repo / ".copier-answers.yml").write_text("_commit: v1\n", encoding="utf-8")
    (pin_repo / ".github/workflows/prek.yaml").write_text(WORKFLOW, encoding="utf-8")
    return pin_repo


def test_check_leaves_the_template_files_to_the_template(
    upgrade: ModuleType, managed_repo: Path, fake: FakeRunner
) -> None:
    """No hook revisions, no pins of shipped workflows; uv waits for the template."""
    items = upgrade.check(fake, managed_repo)
    assert not [i for i in items if i.kind == "hook"]
    assert {i.where for i in items if i.kind == "action"} == {
        ".github/workflows/test.yaml"
    }
    assert [i.protected for i in items if i.kind == "uv"] == [True]


def test_apply_edits_only_what_the_repository_owns(
    upgrade: ModuleType, managed_repo: Path, fake: FakeRunner
) -> None:
    shipped = (managed_repo / ".github/workflows/prek.yaml").read_text()
    config = (managed_repo / ".pre-commit-config.yaml").read_text()
    changed, _ = upgrade.apply(fake, managed_repo)
    assert (managed_repo / ".github/workflows/prek.yaml").read_text() == shipped
    assert (managed_repo / ".pre-commit-config.yaml").read_text() == config
    workflow = (managed_repo / ".github/workflows/test.yaml").read_text()
    assert workflow.count("actions/checkout@v7.1.0") == 2
    assert 'version: "0.12.23"' in workflow  # uv follows the template's workflows
    assert ".github/workflows/prek.yaml" not in changed


def test_the_template_job_upgrades_the_shipped_files(
    upgrade: ModuleType,
    managed_repo: Path,
    fake: FakeRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(upgrade, "INCLUDE_MANAGED", True)
    changed, _ = upgrade.apply(fake, managed_repo)
    assert ".github/workflows/prek.yaml" in changed
    assert "rev: v2.4.3" in (managed_repo / ".pre-commit-config.yaml").read_text()
