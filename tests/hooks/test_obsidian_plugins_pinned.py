"""Every enabled community plugin of this vault is pinned and installed.

@see [[scripts/install_obsidian_plugins.py]]
"""

import importlib.util

from tests.hooks._hookrun import ROOT


def test_every_enabled_repository_plugin_is_pinned_and_installed() -> None:
    """The committed vault ships main.js for every enabled community plugin."""
    spec = importlib.util.spec_from_file_location(
        "install_obsidian_plugins", ROOT / "scripts" / "install_obsidian_plugins.py"
    )
    assert spec is not None
    assert spec.loader is not None
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    for plugin in installer.enabled_plugins():
        assert plugin in installer.REPOSITORIES, plugin
        assert installer.missing(plugin) == [], plugin
