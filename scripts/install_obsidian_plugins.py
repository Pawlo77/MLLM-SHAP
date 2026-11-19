"""Install the vault's Obsidian community plugins from their pinned GitHub releases.

Usage:
    uv run python scripts/install_obsidian_plugins.py [--force | --check | --init]

Downloads ``main.js`` and ``styles.css`` of the release pinned by each committed
``manifest.json`` (Obsidian disables a plugin whose ``main.js`` is missing). Plugin id
to GitHub repository: ``dataview`` and ``realclaudian`` (Claudian) plus the
``[tool.agent-kit.obsidian-plugins]`` table.
"""

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import load as load_config

ROOT = Path(__file__).resolve().parent.parent
OBSIDIAN = ROOT / ".obsidian"


def repositories(root: Path = ROOT) -> dict[str, str]:
    """Plugin id -> GitHub repository, as listed in obsidianmd/obsidian-releases."""
    extra = load_config(root).get("obsidian-plugins", {})
    return {
        "dataview": "blacksmithgu/obsidian-dataview",
        "realclaudian": "YishenTu/claudian",
        **{str(key): str(value) for key, value in extra.items()},
    }


REPOSITORIES: dict[str, str] = repositories()
ASSETS = ("main.js", "styles.css")
Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    """Download ``url`` (GitHub release asset) and return its bytes."""
    # Callers pass https://github.com release URLs built in this module.
    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        return response.read()


def enabled_plugins(obsidian: Path = OBSIDIAN) -> list[str]:
    """Enabled plugin ids, from the example list until the vault has its own."""
    live = obsidian / "community-plugins.json"
    source = live if live.is_file() else obsidian / "community-plugins.example.json"
    return list(json.loads(source.read_text()))


def merge(example: Any, live: Any) -> Any:
    """Return ``live`` with the shared settings of ``example`` applied.

    In a settings object every key of the example wins (hidden folders, enabled
    snippets), keys Obsidian added stay; a plugin list keeps local additions after the
    example's.
    """
    if isinstance(example, dict) and isinstance(live, dict):
        return {**live, **example}
    if isinstance(example, list) and isinstance(live, list):
        return example + [item for item in live if item not in example]
    return example


def seed_settings(obsidian: Path = OBSIDIAN) -> list[str]:
    """Merge each ``*.example.json`` into its live file; return the files that changed.

    Runs on ``make install``, after ``copier update`` and at each agent session start,
    so a fresh clone, a vault Obsidian opened first and a changed example all end up
    with the shared settings. A live file Obsidian left unreadable is replaced.
    """
    seeded: list[str] = []
    for example in sorted(obsidian.rglob("*.example.json")):
        live = example.with_name(example.name.replace(".example.json", ".json"))
        wanted = json.loads(example.read_text())
        try:
            current = json.loads(live.read_text() or "null") if live.is_file() else None
        except json.JSONDecodeError:
            current = None
        merged = wanted if current is None else merge(wanted, current)
        if merged == current:
            continue
        live.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
        seeded.append(str(live.relative_to(obsidian)))
    return seeded


def pinned_version(plugin: str, obsidian: Path = OBSIDIAN) -> str:
    """Return the version pinned by the plugin's committed manifest."""
    manifest = json.loads((obsidian / "plugins" / plugin / "manifest.json").read_text())
    return str(manifest["version"])


def asset_url(repo: str, version: str, asset: str) -> str:
    """Return the download URL of one release asset."""
    return f"https://github.com/{repo}/releases/download/{version}/{asset}"


def missing(plugin: str, obsidian: Path = OBSIDIAN) -> list[str]:
    """Return the asset names the plugin folder lacks (``styles.css`` is optional)."""
    folder = obsidian / "plugins" / plugin
    return [name for name in ("main.js",) if not (folder / name).is_file()]


def install(
    plugin: str,
    *,
    obsidian: Path = OBSIDIAN,
    fetch: Fetch = _fetch,
    force: bool = False,
) -> list[str]:
    """Download the plugin's release assets; return the file names written."""
    repo = REPOSITORIES.get(plugin)
    if repo is None:
        msg = f"{plugin}: no repository pinned in REPOSITORIES"
        raise KeyError(msg)
    version = pinned_version(plugin, obsidian)
    folder = obsidian / "plugins" / plugin
    if not force and not missing(plugin, obsidian):
        return []
    released = json.loads(fetch(asset_url(repo, version, "manifest.json")))
    if str(released.get("version")) != version:
        msg = f"{plugin}: release {version} ships manifest {released.get('version')}"
        raise ValueError(msg)
    written: list[str] = []
    for asset in ASSETS:
        try:
            data = fetch(asset_url(repo, version, asset))
        except Exception as error:
            if asset == "styles.css":
                continue
            msg = f"{plugin}: cannot download {asset} {version}: {error}"
            raise RuntimeError(msg) from error
        (folder / asset).write_bytes(data)
        written.append(asset)
    return written


def main(argv: list[str] | None = None) -> int:
    """Install or check every enabled plugin; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="report, do not download")
    parser.add_argument("--init", action="store_true", help="merge the settings only")
    parser.add_argument("--force", action="store_true", help="re-download everything")
    args = parser.parse_args(argv)
    if args.init:
        print("seeded: " + (", ".join(seed_settings()) or "nothing"))
        return 0
    status = 0
    for plugin in enabled_plugins():
        version = pinned_version(plugin)
        lacking = missing(plugin)
        if args.check:
            state = "missing " + ", ".join(lacking) if lacking else "ok"
            print(f"{plugin} {version}: {state}")
            status |= bool(lacking)
            continue
        written = install(plugin, force=args.force)
        print(
            f"{plugin} {version}: {'wrote ' + ', '.join(written) if written else 'ok'}"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
