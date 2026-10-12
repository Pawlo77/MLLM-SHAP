"""Classify a branch for the delivery autopilot: routine (merges itself) or review.

Usage: ``python scripts/pr_policy.py [--base B] [--json] [--write-marker] [--pr N]``.

Routine: only routine-type commits, nothing under the review paths, the same pin,
``[project]`` changes limited to :func:`_project_review`, within the line and file caps
(without the generated paths), no ``PROTECTED`` file. ``--write-marker`` stores the
result for ``scripts/hooks/merge_gate.sh``; ``--bump`` prints :func:`suggest_bump`.
What differs per repository lives in ``[tool.agent-kit.pr-policy]`` (``Settings``).
Stdlib only.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess  # nosec B404 - fixed git argv, no shell
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import load as load_config

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

MAX_LINES = 400
"""Most changed lines (insertions plus deletions) a routine branch may have."""

MAX_FILES = 25
"""Most changed files a routine branch may have."""

ROUTINE_TYPES = frozenset({"chore", "ci", "build", "docs", "test"})
"""Default commit subject types that may merge without the maintainer."""

PROTECTED = frozenset({
    ".claude/settings.json",
    ".codex/config.toml",
    ".codex/hooks.json",
    ".cursor/hooks.json",
    ".github/workflows/template-drift.yaml",
    ".github/workflows/template-sync.yaml",
    "scripts/pr_policy.py",
    "scripts/template_drift.py",
    "scripts/hooks/merge_gate.sh",
    "scripts/hooks/commit_msg.py",
    "scripts/hooks/lib.sh",
    "scripts/hooks/gate_lib.sh",
    "scripts/hooks/shell_guard.py",
    "scripts/hooks/shell_guard.sh",
})
"""Security-relevant files: a change to any of them always gets a human merge.

A repository adds its own (a publish workflow, say) under ``protected``."""

PROJECT_KEYS = ("version", "dependencies", "optional-dependencies")
"""``[project]`` keys whose change needs the maintainer's review."""

DEPENDENCY_TYPES = frozenset({"build", "chore", "ci"})
"""Commit types under which a dependency specifier change stays routine."""

GENERATED = ("uv.lock",)
"""Default generated paths, left out of the line and file caps."""

MAIN_BRANCHES = frozenset({"main", "master"})
"""Branches the autopilot never writes a marker for."""

BUMPS = ("none", "patch", "minor", "major")
"""Version bump levels of :func:`suggest_bump`, lowest first."""

_SUBJECT = re.compile(r"^(?P<type>[a-z]+)(?:\([^)]*\))?(?P<bang>!)?: \S")
_RELEASE = re.compile(r"^chore: release v(?P<version>\d+\.\d+\.\d+)$")
_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_SHORTSTAT = re.compile(r"(\d+) (insertion|deletion)")

Kind = Literal["routine", "review"]


@dataclass(frozen=True)
class Settings:
    """Per-repository policy, read from ``[tool.agent-kit.pr-policy]``."""

    routine_types: frozenset[str] = ROUTINE_TYPES
    review_paths: tuple[str, ...] = ()
    protected: frozenset[str] = PROTECTED
    generated: tuple[str, ...] = GENERATED
    pin: str = ""
    release: bool = False
    max_lines: int = MAX_LINES
    max_files: int = MAX_FILES


def settings(root: Path = ROOT) -> Settings:
    """Return the policy of the repository at ``root`` (defaults when unset).

    Keys: ``routine-types`` (words), ``review-paths`` and ``generated`` (glob lists),
    ``protected`` (extra files), ``pin`` (a locked package whose change needs review),
    ``release`` (a ``chore: release vX.Y.Z`` commit may change the version),
    ``max-lines`` and ``max-files``.
    """
    raw = load_config(root).get("pr-policy", {})
    table = raw if isinstance(raw, dict) else {}
    defaults = Settings()
    return Settings(
        routine_types=frozenset(
            str(table.get("routine-types", " ".join(ROUTINE_TYPES))).split()
        ),
        review_paths=tuple(table.get("review-paths", defaults.review_paths)),
        protected=PROTECTED | frozenset(table.get("protected", ())),
        generated=tuple(table.get("generated", defaults.generated)),
        pin=str(table.get("pin", "")),
        release=bool(table.get("release", False)),
        max_lines=int(table.get("max-lines", MAX_LINES)),
        max_files=int(table.get("max-files", MAX_FILES)),
    )


class PolicyError(RuntimeError):
    """The classification or the marker cannot be produced."""


class Marker(TypedDict):
    """Content of ``.remember/tmp/pr-autopilot.<session>.json``."""

    kind: Kind
    head: str
    branch: str
    pr: int | None
    created: str
    base: str
    diff: str


@dataclass(frozen=True)
class Classification:
    """Kind of a branch and the reasons for it."""

    kind: Kind
    reasons: tuple[str, ...]
    files: tuple[str, ...]


def _git(root: Path, *args: str) -> str:
    """Run git in ``root`` and return its stripped output.

    Raises:
        PolicyError: When git fails.
    """
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed git argv
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PolicyError(f"git {' '.join(args)} failed") from exc  # noqa: TRY003
    return proc.stdout.strip()


def _project(root: Path, rev: str) -> dict[str, object]:
    """The ``[project]`` table of ``pyproject.toml`` at ``rev`` (empty when absent)."""
    try:
        text = _git(root, "show", f"{rev}:pyproject.toml")
        table = tomllib.loads(text).get("project", {})
    except (PolicyError, tomllib.TOMLDecodeError):
        return {}
    return cast("dict[str, object]", table) if isinstance(table, dict) else {}


def _packages(value: object) -> object:
    """Normalised package names of a requirement list, or of each optional extra."""
    if isinstance(value, dict):
        return {key: _packages(item) for key, item in value.items()}
    if not isinstance(value, list):
        return value
    names = [_REQUIREMENT.match(str(item)) for item in value]
    return sorted(re.sub(r"[-_.]+", "-", m[1]).lower() if m else "" for m in names)


def _project_review(
    old: dict[str, object],
    new: dict[str, object],
    subjects: list[str],
    release: bool = False,
) -> list[str]:
    """``[project]`` keys whose change needs the maintainer.

    A dependency change keeps the package set and has only ``DEPENDENCY_TYPES``
    commits; a version change always needs the maintainer, unless ``release`` is set
    and it equals the version of a ``chore: release vX.Y.Z`` commit.
    """
    types = {
        m["type"] + (m["bang"] or "") if m else ""
        for m in map(_SUBJECT.match, subjects)
    }
    releases = {m["version"] for m in map(_RELEASE.match, subjects) if m and release}
    keys = []
    for key in PROJECT_KEYS:
        if old.get(key) == new.get(key):
            continue
        if key == "version" and new.get(key) in releases:
            continue
        same = _packages(old.get(key)) == _packages(new.get(key))
        if key != "version" and same and types <= DEPENDENCY_TYPES:
            continue
        keys.append(key)
    return keys


def _pinned(root: Path, rev: str, pin: str) -> str:
    """The locked entry of package ``pin`` in ``uv.lock`` at ``rev`` ("" if absent)."""
    name = re.escape(pin)
    pattern = rf'(?ms)^\[\[package\]\]\nname = "{name}"\n(.*?)(?=^\[\[|\Z)'
    try:
        match = re.search(pattern, _git(root, "show", f"{rev}:uv.lock"))
    except PolicyError:
        return ""
    return match[1] if match else ""


def _counted(files: list[str], generated: Sequence[str] = GENERATED) -> list[str]:
    return [f for f in files if not any(fnmatch.fnmatchcase(f, g) for g in generated)]


def _names(root: Path, base: str) -> list[str]:
    out = _git(root, "diff", "--name-only", f"{base}...HEAD")
    return [line for line in out.splitlines() if line]


def diff_hash(base: str, root: Path = ROOT) -> str:
    """SHA-256 of the changed path list of ``<base>...HEAD``."""
    return hashlib.sha256("\n".join(_names(root, base)).encode()).hexdigest()


def classify(base: str = "main", root: Path = ROOT) -> Classification:
    """Classify the commits of ``HEAD`` that are not on ``base``.

    Args:
        base: Branch or revision the pull request merges into.
        root: Repository root.

    Returns:
        ``routine`` when every rule in the module docstring holds, else ``review``
        with one reason per broken rule.
    """
    cfg = settings(root)
    files = _names(root, base)
    subjects = _git(root, "log", "--format=%s", f"{base}..HEAD").splitlines()
    review: list[str] = []
    if not subjects:
        review.append("no commits on the branch")
    types: list[str] = []
    for subject in subjects:
        match = _SUBJECT.match(subject)
        kind = (match["type"] + (match["bang"] or "")) if match else subject[:40]
        types.append(kind)
        if not match or match["bang"] or match["type"] not in cfg.routine_types:
            review.append(f"commit type {kind!r} is not routine")
    hits = {
        p: [f for f in files if fnmatch.fnmatchcase(f, p)] for p in cfg.review_paths
    }
    review += [
        f"{len(h)} changed files under {p.removesuffix('*')}"
        for p, h in hits.items()
        if h
    ]
    merge_base = _git(root, "merge-base", base, "HEAD")
    if "pyproject.toml" in files:
        old, new = _project(root, merge_base), _project(root, "HEAD")
        if changed := _project_review(old, new, subjects, cfg.release):
            review.append(f"pyproject.toml [project] {', '.join(changed)} changed")
    pin_moved = cfg.pin and "uv.lock" in files
    if pin_moved and _pinned(root, merge_base, cfg.pin) != _pinned(
        root, "HEAD", cfg.pin
    ):
        review.append(f"the {cfg.pin} pin in uv.lock changed (results may change)")
    review += [f"{f} is security-relevant" for f in files if f in cfg.protected]
    excluded = [f":(exclude,glob){pattern}" for pattern in cfg.generated]
    stat = _git(root, "diff", "--shortstat", f"{base}...HEAD", "--", ".", *excluded)
    lines = sum(int(n) for n, _ in _SHORTSTAT.findall(stat))
    if lines > cfg.max_lines:
        review.append(f"{lines} changed lines (cap {cfg.max_lines})")
    if (counted := len(_counted(files, cfg.generated))) > cfg.max_files:
        review.append(f"{counted} files (cap {cfg.max_files})")
    if review:
        return Classification("review", tuple(review), tuple(files))
    reasons = (
        f"commit types: {', '.join(sorted(set(types)))}",
        (
            "nothing under the review paths, no reviewed [project] or pin change, "
            "no protected file"
        ),
        f"{len(files)} files, {lines} changed lines",
    )
    return Classification("routine", reasons, tuple(files))


def _semver(text: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in text.removeprefix("v").split("."))
    return major, minor, patch


def suggest_bump(root: Path = ROOT) -> tuple[str, str]:
    """Bump the commits since the last ``v*`` tag call for, and the next version.

    ``!`` or ``BREAKING CHANGE`` is major (minor below 1.0), ``feat`` minor, ``fix`` and
    ``perf`` patch. The next version never falls below the ``pyproject.toml`` one, so
    a branch that already carries the release commit gets the same answer.
    """
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    current = _semver(str(version["project"]["version"]))
    try:
        tag = _git(root, "describe", "--tags", "--abbrev=0", "--match", "v*")
    except PolicyError:
        tag = ""
    log = _git(
        root, "log", "--no-merges", "--format=%B%x00", f"{tag}..HEAD" if tag else "HEAD"
    )
    level = 0
    for message in filter(None, (m.strip() for m in log.split("\0"))):
        match = _SUBJECT.match(message)
        if (match and match["bang"]) or "BREAKING CHANGE" in message:
            level = 3
        elif match and match["type"] == "feat":
            level = max(level, 2)
        elif match and match["type"] in {"fix", "perf"}:
            level = max(level, 1)
    major, minor, patch = base = _semver(tag) if tag else current
    level = 2 if level == 3 and major == 0 else level
    bumped = [base, (major, minor, patch + 1), (major, minor + 1, 0), (major + 1, 0, 0)]
    nxt = max(bumped[level], current) if level else current
    return BUMPS[level], ".".join(map(str, nxt))


def session_key(explicit: str | None = None) -> str:
    """Session key as ``gate_lib.sh`` derives it: [A-Za-z0-9_-], 64 characters.

    Raises:
        PolicyError: When neither ``explicit`` nor ``CLAUDE_CODE_SESSION_ID`` is set.
    """
    raw = explicit or os.environ.get("CLAUDE_CODE_SESSION_ID", "")
    key = re.sub(r"[^A-Za-z0-9_-]", "", raw)[:64]
    if not key:
        raise PolicyError(  # noqa: TRY003
            "no session key: pass --session with the key the session-start hook printed"
        )
    return key


def marker_path(key: str, root: Path = ROOT) -> Path:
    """Path of the marker for session ``key``."""
    return root / ".remember" / "tmp" / f"pr-autopilot.{key}.json"


def _read(path: Path) -> Marker | None:
    """Parse a marker file; None when missing or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    text = ("head", "branch", "created", "base", "diff")
    if data.get("kind") not in ("routine", "review"):
        return None
    if not all(isinstance(data.get(k), str) and data[k] for k in text):
        return None
    pr = data.get("pr")
    if pr is not None and (not isinstance(pr, int) or isinstance(pr, bool)):
        return None
    return cast("Marker", data)


def _save(marker: Marker, key: str, root: Path) -> None:
    path = marker_path(key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def write_marker(
    base: str, key: str, root: Path = ROOT, pr: int | None = None
) -> Marker:
    """Classify the branch and write the marker for session ``key``.

    Args:
        base: Branch the pull request merges into.
        key: Session key (:func:`session_key`).
        root: Repository root.
        pr: Pull request number; when None, the number of an earlier marker for the
            same branch is kept.

    Returns:
        The marker written.

    Raises:
        PolicyError: On ``main``/``master`` or a detached ``HEAD``.
    """
    branch = _git(root, "branch", "--show-current")
    if not branch or branch in MAIN_BRANCHES:
        raise PolicyError(f"not on a feature branch ({branch or 'detached HEAD'})")  # noqa: TRY003
    previous = _read(marker_path(key, root))
    if pr is None and previous and previous["branch"] == branch:
        pr = previous["pr"]
    marker: Marker = {
        "kind": classify(base, root).kind,
        "head": _git(root, "rev-parse", "HEAD"),
        "branch": branch,
        "pr": pr,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "base": base,
        "diff": diff_hash(base, root),
    }
    _save(marker, key, root)
    return marker


def set_pr(key: str, pr: int, root: Path = ROOT) -> Marker:
    """Record the pull request number in the marker of session ``key``.

    Raises:
        PolicyError: When the session has no marker.
    """
    marker = _read(marker_path(key, root))
    if marker is None:
        raise PolicyError("no marker for this session: run --write-marker first")  # noqa: TRY003
    marker["pr"] = pr
    _save(marker, key, root)
    return marker


def load_valid_marker(key: str, root: Path = ROOT) -> tuple[Marker, bool] | None:
    """Return the marker of session ``key`` while the classification is current.

    Returns:
        The marker and whether its ``head`` is ``HEAD`` itself, or None when there is
        no marker, the branch differs, ``head`` is not an ancestor of ``HEAD`` or the
        changed path list differs from the classified one.
    """
    marker = _read(marker_path(key, root))
    if marker is None:
        return None
    try:
        branch = _git(root, "branch", "--show-current")
        if branch != marker["branch"] or branch in MAIN_BRANCHES:
            return None
        head = _git(root, "rev-parse", "HEAD")
        if marker["head"] != head:
            _git(root, "merge-base", "--is-ancestor", marker["head"], "HEAD")
        if diff_hash(marker["base"], root) != marker["diff"]:
            return None
    except PolicyError:
        return None
    return marker, marker["head"] == head


def main(argv: Sequence[str] | None = None) -> int:
    """Print the classification; write or update the session marker on request."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="main")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--write-marker", action="store_true")
    parser.add_argument("--pr", type=int, help="record the pull request number")
    parser.add_argument("--session", help="session key (default from the agent)")
    parser.add_argument(
        "--merge-ok",
        action="store_true",
        help="print 'ok' when this session's marker is routine, current and has a PR",
    )
    parser.add_argument("--bump", action="store_true", help="print the next version")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.bump:
            print(" ".join(suggest_bump(args.root)))
            return 0
        if args.merge_ok:
            loaded = load_valid_marker(session_key(args.session), args.root)
            if loaded is None:
                print("no current marker (run --write-marker after make check)")
                return 1
            marker, exact = loaded
            if marker["kind"] == "routine" and exact and marker["pr"] is not None:
                print("ok")
                return 0
            print(f"marker is {marker['kind']}, exact={exact}, pr={marker['pr']}")
            return 1
        if args.pr is not None and not args.write_marker:
            marker = set_pr(session_key(args.session), args.pr, args.root)
            print(f"marker: {marker['kind']}, pr {marker['pr']}")
            return 0
        result = classify(args.base, args.root)
        if args.write_marker:
            write_marker(args.base, session_key(args.session), args.root, args.pr)
    except PolicyError as exc:
        print(f"pr_policy: {exc}", file=sys.stderr)
        return 2
    if args.json:
        payload = {"kind": result.kind, "reasons": list(result.reasons)}
        print(json.dumps({**payload, "files": list(result.files)}, indent=2))
    else:
        print(result.kind)
        print("\n".join(f"- {reason}" for reason in result.reasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
