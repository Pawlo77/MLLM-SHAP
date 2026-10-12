"""Tests for the pinned Obsidian plugin installer (offline, fake fetch).

@see [[scripts/install_obsidian_plugins.py]]
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT


@pytest.fixture
def installer() -> ModuleType:
    """Load scripts/install_obsidian_plugins.py."""
    spec = importlib.util.spec_from_file_location(
        "install_obsidian_plugins", ROOT / "scripts" / "install_obsidian_plugins.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A vault with one enabled plugin whose manifest pins version 1.2.3."""
    obsidian = tmp_path / ".obsidian"
    (obsidian / "plugins" / "dataview").mkdir(parents=True)
    (obsidian / "community-plugins.json").write_text(json.dumps(["dataview"]))
    (obsidian / "plugins" / "dataview" / "manifest.json").write_text(
        json.dumps({"id": "dataview", "version": "1.2.3"})
    )
    return obsidian


def test_asset_url_points_at_the_pinned_release(installer) -> None:
    url = installer.asset_url("blacksmithgu/obsidian-dataview", "0.5.68", "main.js")
    assert url == (
        "https://github.com/blacksmithgu/obsidian-dataview/releases/download/0.5.68/main.js"
    )


def test_install_writes_main_js_and_styles(installer, vault: Path) -> None:
    served = {
        "manifest.json": json.dumps({"id": "dataview", "version": "1.2.3"}).encode(),
        "main.js": b"console.log('dataview')",
        "styles.css": b".dataview {}",
    }

    def fetch(url: str) -> bytes:
        return served[url.rsplit("/", 1)[1]]

    written = installer.install("dataview", obsidian=vault, fetch=fetch)
    assert written == ["main.js", "styles.css"]
    assert (vault / "plugins" / "dataview" / "main.js").read_bytes() == served[
        "main.js"
    ]
    assert installer.missing("dataview", vault) == []
    assert installer.install("dataview", obsidian=vault, fetch=fetch) == []


def test_install_refuses_a_release_with_another_version(installer, vault: Path) -> None:
    def fetch(url: str) -> bytes:
        if url.endswith("manifest.json"):
            return json.dumps({"id": "dataview", "version": "9.9.9"}).encode()
        return b""

    with pytest.raises(ValueError, match=r"ships manifest 9\.9\.9"):
        installer.install("dataview", obsidian=vault, fetch=fetch)


def test_styles_css_is_optional(installer, vault: Path) -> None:
    def fetch(url: str) -> bytes:
        name = url.rsplit("/", 1)[1]
        if name == "manifest.json":
            return json.dumps({"id": "dataview", "version": "1.2.3"}).encode()
        if name == "styles.css":
            raise OSError("404")
        return b"main"

    assert installer.install("dataview", obsidian=vault, fetch=fetch) == ["main.js"]


def test_repositories_merge_the_project_table(installer, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.agent-kit.obsidian-plugins]\n"
        'templater-obsidian = "silentvoid13/Templater"\n',
        encoding="utf-8",
    )
    found = installer.repositories(tmp_path)
    assert found["dataview"] == "blacksmithgu/obsidian-dataview"
    assert found["templater-obsidian"] == "silentvoid13/Templater"
    assert installer.repositories(tmp_path / "missing") == {
        "dataview": "blacksmithgu/obsidian-dataview"
    }
