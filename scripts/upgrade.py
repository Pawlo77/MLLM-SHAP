"""List or apply upgrades of packages, prek hooks, action pins, uv and Obsidian plugins.

``--check`` prints one line per outdated item and exits 1 when there is any; ``--apply``
upgrades every item that is not protected and prints the changed files. Protected items
are reported, never edited: the ``protected-workflows`` (their pins and the uv version)
and the package ``pin``, which only ``--pin`` moves, on its own review branch. Both come
from ``[tool.agent-kit.upgrade]``. The files the template ships (:data:`KIT_WORKFLOWS`,
the prek config) are bumped in the template; its own job passes ``--include-managed``.
External commands go through an injectable :data:`Runner`. Stdlib only.
"""

import argparse
import re
import shutil
import subprocess  # nosec B404 - fixed argv lists, no shell
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import load as load_config

try:
    from install_obsidian_plugins import REPOSITORIES, pinned_version
except ImportError:  # a repository without the Obsidian plugin installer
    REPOSITORIES: dict[str, str] = {}  # type: ignore[no-redef]
    pinned_version = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

SYNC_ARGS = "--all-groups"
"""Default ``uv sync`` arguments after an upgrade."""

KIT_WORKFLOWS = frozenset({
    "assign-author.yaml",
    "codeql.yaml",
    "pr-title.yaml",
    "prek.yaml",
    "template-drift.yaml",
    "template-sync.yaml",
})
"""Workflows the template ships; their pins, like the hook revisions, move there."""

PRE_COMMIT = ".pre-commit-config.yaml"
PREK = ("uv", "run", "prek", "update", "--color", "never", "--no-progress")
INSTALL_PLUGINS = ("uv", "run", "python", "scripts/install_obsidian_plugins.py")

_LOCK = re.compile(r"^(?:Would )?[Uu]pdate (\S+) v?(\S+) -> v?(\S+)$", re.MULTILINE)
_HOOK = re.compile(r"would update rev `([^`]+)` -> `([^`]+)`")
_PIN = re.compile(
    r"(?P<pre>uses:[ \t]*)(?P<action>[\w.-]+/[\w.-]+[\w./-]*)@(?P<ref>[\w.-]+)"
    r"(?:[ \t]*#[ \t]*(?P<tag>v?\d[\w.-]*))?"
)
_SHA = re.compile(r"[0-9a-f]{40}")
_REQUIRED = re.compile(r'(required-version = ">=)([^"]+)(")')
_UV_PIN = re.compile(r'(?m)^([ \t]+version: )"(\d[\w.]*)"$')
_VERSION = re.compile(r"^v?(\d+(?:\.\d+)*)$")

Runner = Callable[[Sequence[str], Path], str]
"""Run ``argv`` in a directory and return stdout plus stderr; raise on failure."""


class UpgradeError(RuntimeError):
    """An external command failed."""


@dataclass(frozen=True)
class Settings:
    """Per-repository settings, read from ``[tool.agent-kit.upgrade]``.

    ``protected-workflows``: workflow file names whose pins are report-only;
    ``pin``: a locked package only ``--pin`` moves, ``pin-git``: the name of the git
    repository behind it (as in ``.../NAME.git``); ``sync-args``: ``uv sync`` arguments.
    """

    protected_workflows: tuple[str, ...] = ()
    pin: str = ""
    pin_git: str = ""
    sync_args: tuple[str, ...] = tuple(SYNC_ARGS.split())


def settings(root: Path = ROOT) -> Settings:
    """Return the settings of the repository at ``root`` (defaults when unset)."""
    raw = load_config(root).get("upgrade", {})
    table = raw if isinstance(raw, dict) else {}
    return Settings(
        protected_workflows=tuple(table.get("protected-workflows", ())),
        pin=str(table.get("pin", "")),
        pin_git=str(table.get("pin-git", "")),
        sync_args=tuple(str(table.get("sync-args", SYNC_ARGS)).split()),
    )


@dataclass(frozen=True)
class Item:
    """One outdated pin."""

    kind: str
    name: str
    old: str
    new: str
    where: str
    protected: bool = False

    def __str__(self) -> str:
        """One report line."""
        note = "  (protected: report only)" if self.protected else ""
        return (
            f"{self.kind:<8} {self.name} {self.old} -> {self.new}  {self.where}{note}"
        )


def run(argv: Sequence[str], cwd: Path) -> str:
    """Default :data:`Runner`: subprocess without a shell."""
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv lists
            list(argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UpgradeError(f"{' '.join(argv[:4])} failed: {exc}") from exc  # noqa: TRY003
    if proc.returncode:
        tail = (proc.stderr or proc.stdout).strip()[-400:]
        raise UpgradeError(f"{' '.join(argv[:4])} failed: {tail}")  # noqa: TRY003
    return proc.stdout + proc.stderr


def _version(tag: str) -> tuple[int, ...] | None:
    match = _VERSION.match(tag)
    return tuple(int(part) for part in match[1].split(".")) if match else None


def newer(old: str, new: str) -> bool:
    """Whether ``new`` is a higher version than ``old`` (both like ``v1.2.3``)."""
    old_v, new_v = _version(old), _version(new)
    return old_v is not None and new_v is not None and new_v > old_v


class GitHub:
    """Latest release tags and tag commits through ``gh api`` GET calls, cached."""

    def __init__(self, runner: Runner, root: Path) -> None:
        """Bind the runner and the directory ``gh`` runs in."""
        self._run, self._root = runner, root
        self._latest: dict[str, str] = {}

    def _api(self, *args: str) -> str:
        return self._run(["gh", "api", *args], self._root).strip()

    def latest(self, repo: str) -> str:
        """Tag of the latest release, else the highest ``vX.Y.Z`` tag."""
        if repo not in self._latest:
            try:
                tag = self._api(f"repos/{repo}/releases/latest", "--jq", ".tag_name")
            except UpgradeError:
                tag = ""
            if _version(tag) is None:
                refs = self._api(
                    "--paginate",
                    f"repos/{repo}/git/matching-refs/tags/v",
                    "--jq",
                    ".[].ref",
                ).split()
                tags = [r.removeprefix("refs/tags/") for r in refs]
                tags = [t for t in tags if _version(t) and t.count(".") == 2]
                tag = max(tags, key=lambda t: _version(t) or (), default="")
            self._latest[repo] = tag
        return self._latest[repo]

    def sha(self, repo: str, tag: str) -> str:
        """Commit sha of ``tag`` (annotated tags dereferenced)."""
        kind, sha = self._api(
            f"repos/{repo}/git/ref/tags/{tag}",
            "--jq",
            '.object.type + " " + .object.sha',
        ).split()
        if kind == "tag":
            sha = self._api(f"repos/{repo}/git/tags/{sha}", "--jq", ".object.sha")
        return sha


INCLUDE_MANAGED = False
"""Set by ``--include-managed``: the template's own job upgrades the shipped files."""


def _managed(root: Path) -> bool:
    """Whether the template ships the workflows and prek config of ``root``."""
    return not INCLUDE_MANAGED and (root / ".copier-answers.yml").is_file()


def _own(root: Path, path: Path) -> bool:
    """Whether ``path`` is a workflow this repository maintains itself."""
    return not (_managed(root) and path.name in KIT_WORKFLOWS)


def python_deps(runner: Runner, root: Path, _gh: GitHub) -> list[Item]:
    """Locked packages ``uv lock --upgrade`` would update; the pin is protected."""
    cfg = settings(root)
    out = runner(["uv", "lock", "--upgrade", "--dry-run"], root)
    items = [
        Item("python", n, o, v, "uv.lock", protected=bool(cfg.pin) and n == cfg.pin)
        for n, o, v in _LOCK.findall(out)
    ]
    if cfg.pin and cfg.pin_git and (root / "uv.lock").is_file():
        git = re.escape(cfg.pin_git)
        moved = re.search(rf"Updated \S*/{git}\.git \((?P<sha>[0-9a-f]{{40}})\)", out)
        text = (root / "uv.lock").read_text(encoding="utf-8")
        locked = re.search(
            rf'(?m)^source = \{{ git = "\S*/{git}\.git\S*#(?P<sha>\w+)', text
        )
        listed = any(item.name == cfg.pin for item in items)
        if locked and moved and locked["sha"] != moved["sha"] and not listed:
            old, new = locked["sha"][:7], moved["sha"][:7]
            items.append(Item("python", cfg.pin, old, new, "uv.lock", protected=True))
    return items


def _prek(runner: Runner, root: Path, *, dry_run: bool) -> tuple[str, str]:
    """Run prek's updater on a copy of the config (prek refuses an unstaged one)."""
    config = root / PRE_COMMIT
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / config.name
        shutil.copyfile(config, copy)
        flags = ["--dry-run"] if dry_run else []
        out = runner([*PREK, *flags, "-c", str(copy)], root)
        return out, copy.read_text(encoding="utf-8")


def hook_revs(runner: Runner, root: Path, _gh: GitHub) -> list[Item]:
    """Prek hook repositories with a newer tagged revision (none when managed)."""
    if _managed(root):
        return []
    out, _ = _prek(runner, root, dry_run=True)
    items, repo = [], ""
    for line in out.splitlines():
        if line and not line[0].isspace():
            repo = line.strip()
        elif match := _HOOK.search(line):
            old, new = match.groups()
            items.append(Item("hook", repo, old, new, PRE_COMMIT))
    return items


def _workflows(root: Path) -> list[Path]:
    return sorted((root / ".github" / "workflows").glob("*.y*ml"))


def _pin(match: re.Match[str]) -> tuple[str, str] | None:
    """Owner/repo and version tag of one action pin; None when it has no version."""
    ref = match["ref"]
    tag = match["tag"] if _SHA.fullmatch(ref) else ref
    if not tag or _version(tag) is None:
        return None
    return "/".join(match["action"].split("/")[:2]), tag


def _pins(text: str) -> Iterator[tuple[re.Match[str], str, str]]:
    """Each versioned action pin with its owner/repo and tag."""
    for match in _PIN.finditer(text):
        if pin := _pin(match):
            yield match, *pin


def action_pins(_runner: Runner, root: Path, gh: GitHub) -> list[Item]:
    """Workflow action pins below the action's latest release."""
    items: dict[tuple[str, str, str], Item] = {}
    for path in filter(lambda p: _own(root, p), _workflows(root)):
        for match, repo, tag in _pins(path.read_text(encoding="utf-8")):
            latest = gh.latest(repo)
            if newer(tag, latest):
                where = path.relative_to(root).as_posix()
                protected = path.name in settings(root).protected_workflows
                item = Item("action", match["action"], tag, latest, where, protected)
                items[where, item.name, tag] = item
    return list(items.values())


def _protected_pins_uv(root: Path) -> bool:
    """Whether a workflow outside this repository's control pins uv.

    Then ``required-version`` stays put: it must not outrun the version those pin.
    """
    if _managed(root) and any(
        (root / ".github" / "workflows" / name).is_file() for name in KIT_WORKFLOWS
    ):
        return True
    return any(
        path.is_file() and bool(_UV_PIN.search(path.read_text(encoding="utf-8")))
        for path in (
            root / ".github" / "workflows" / name
            for name in settings(root).protected_workflows
        )
    )


def uv_version(_runner: Runner, root: Path, gh: GitHub) -> list[Item]:
    """``required-version`` and the workflow uv pins below uv's latest release."""
    match = _REQUIRED.search((root / "pyproject.toml").read_text(encoding="utf-8"))
    if match is None:
        return []
    latest = gh.latest("astral-sh/uv")
    if not newer(match[2], latest):
        return []
    where = "pyproject.toml, .github/workflows"
    return [Item("uv", "uv", match[2], latest, where, _protected_pins_uv(root))]


def obsidian_plugins(_runner: Runner, root: Path, gh: GitHub) -> list[Item]:
    """Pinned Obsidian plugin manifests below the plugin's latest release."""
    items = []
    for plugin, repo in REPOSITORIES.items():
        if pinned_version is None:
            break
        manifest = root / ".obsidian" / "plugins" / plugin / "manifest.json"
        if not manifest.is_file():
            continue
        pinned, latest = pinned_version(plugin, root / ".obsidian"), gh.latest(repo)
        if newer(pinned, latest):
            where = manifest.relative_to(root).as_posix()
            items.append(Item("obsidian", plugin, pinned, latest, where))
    return items


CHECKS = (python_deps, hook_revs, action_pins, uv_version, obsidian_plugins)
"""Every check, in report order."""


def check(runner: Runner = run, root: Path = ROOT) -> list[Item]:
    """Every outdated item."""
    gh = GitHub(runner, root)
    return [item for found in CHECKS for item in found(runner, root, gh)]


def _repin(match: re.Match[str], gh: GitHub) -> str:
    pin = _pin(match)
    latest = gh.latest(pin[0]) if pin else ""
    if pin is None or not newer(pin[1], latest):
        return match[0]
    if _SHA.fullmatch(match["ref"]):
        return f"{match['pre']}{match['action']}@{gh.sha(pin[0], latest)} # {latest}"
    return f"{match['pre']}{match['action']}@{latest}"


def _write(path: Path, text: str) -> None:
    if path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _apply_pins(root: Path, gh: GitHub) -> None:
    for path in _workflows(root):
        if _own(root, path) and path.name not in settings(root).protected_workflows:
            text = path.read_text(encoding="utf-8")
            _write(path, _PIN.sub(lambda match: _repin(match, gh), text))


def _apply_uv(item: Item, root: Path) -> None:
    pyproject = root / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    _write(pyproject, _REQUIRED.sub(rf"\g<1>{item.new}\g<3>", text))
    for path in filter(lambda p: _own(root, p), _workflows(root)):
        text = path.read_text(encoding="utf-8")
        _write(path, _UV_PIN.sub(rf'\g<1>"{item.new}"', text))


def _apply_plugin(item: Item, root: Path) -> None:
    manifest = root / item.where
    pattern = rf'("version"\s*:\s*"){re.escape(item.old)}"'
    text = manifest.read_text(encoding="utf-8")
    _write(manifest, re.sub(pattern, rf'\g<1>{item.new}"', text, count=1))


def _snapshot(root: Path) -> dict[Path, bytes]:
    """Contents of every file an upgrade may change."""
    watched = [root / name for name in ("pyproject.toml", "uv.lock", PRE_COMMIT)]
    watched += _workflows(root) + sorted(root.glob(".obsidian/plugins/*/manifest.json"))
    return {p: p.read_bytes() for p in watched if p.is_file()}


def _changed(root: Path, before: dict[Path, bytes]) -> list[str]:
    return [
        p.relative_to(root).as_posix()
        for p in before
        if p.is_file() and p.read_bytes() != before[p]
    ]


def apply(runner: Runner = run, root: Path = ROOT) -> tuple[list[str], list[Item]]:
    """Upgrade everything that is not protected.

    Returns:
        The changed files and the protected items left for the maintainer.
    """
    cfg = settings(root)
    gh = GitHub(runner, root)
    before = _snapshot(root)
    deps = python_deps(runner, root, gh)
    protected = [item for item in deps if item.protected]
    if not cfg.pin:
        runner(["uv", "lock", "--upgrade"], root)
        runner(["uv", "sync", *cfg.sync_args], root)
    elif names := [item.name for item in deps if not item.protected]:
        runner(["uv", "lock", *(f"--upgrade-package={name}" for name in names)], root)
        runner(["uv", "sync", *cfg.sync_args], root)
    if not _managed(root):
        _write(root / PRE_COMMIT, _prek(runner, root, dry_run=False)[1])
    protected += [item for item in action_pins(runner, root, gh) if item.protected]
    _apply_pins(root, gh)
    for item in uv_version(runner, root, gh):
        if item.protected:
            protected.append(item)
        else:
            _apply_uv(item, root)
    plugins = obsidian_plugins(runner, root, gh)
    for item in plugins:
        _apply_plugin(item, root)
    if plugins:
        runner([*INSTALL_PLUGINS, "--force"], root)
    return _changed(root, before), protected


def move_pin(runner: Runner = run, root: Path = ROOT) -> list[str]:
    """Move the configured pin to its branch head; return the changed files.

    Raises:
        UpgradeError: When no ``pin`` is configured.
    """
    cfg = settings(root)
    if not cfg.pin:
        raise UpgradeError("no pin configured in [tool.agent-kit.upgrade]")  # noqa: TRY003
    before = _snapshot(root)
    runner(["uv", "lock", f"--upgrade-package={cfg.pin}"], root)
    runner(["uv", "sync", *cfg.sync_args], root)
    return _changed(root, before)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``--check`` or ``--apply``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="list outdated items")
    mode.add_argument("--apply", action="store_true", help="upgrade them")
    mode.add_argument("--pin", action="store_true", help="move the configured pin")
    parser.add_argument(
        "--include-managed",
        action="store_true",
        help="also upgrade the files the template ships (the template's own job)",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    global INCLUDE_MANAGED
    INCLUDE_MANAGED = args.include_managed
    try:
        if args.check:
            items = check(run, args.root)
            print("\n".join(map(str, items)) if items else "everything is current")
            return int(bool(items))
        protected: list[Item] = []
        if args.pin:
            changed = move_pin(run, args.root)
        else:
            changed, protected = apply(run, args.root)
    except UpgradeError as exc:
        print(f"upgrade: {exc}", file=sys.stderr)
        return 2
    print("\n".join(f"changed  {path}" for path in changed) or "nothing changed")
    print("\n".join(str(item) for item in protected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
