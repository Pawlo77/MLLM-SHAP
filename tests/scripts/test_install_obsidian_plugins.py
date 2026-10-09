"""Tests for the pinned Obsidian plugin installer (offline, fake fetch).

@domain tooling
@see [[scripts/install_obsidian_plugins.py]]
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def installer(script):
    """Load scripts/install_obsidian_plugins.py."""
    return script("install_obsidian_plugins")


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


def test_every_enabled_repository_plugin_is_pinned_and_installed(installer) -> None:
    """The committed vault ships main.js for every enabled community plugin."""
    for plugin in installer.enabled_plugins():
        assert plugin in installer.REPOSITORIES, plugin
        assert installer.missing(plugin) == [], plugin
