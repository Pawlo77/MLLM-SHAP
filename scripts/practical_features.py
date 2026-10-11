"""Print the practical-test feature ids a branch's changed paths map to.

Usage: ``python scripts/practical_features.py [--base main] [--root .]``. Changed
paths (commits since the merge base, working-tree edits, untracked files) go
through ``tests/practical/feature_map.yaml``: every glob a path matches adds its
ids, ``smoke`` adds the smoke set and ``none`` adds nothing. Prints the sorted ids
comma-separated, an empty line when there are none; skill ``ship`` passes them to
``make practical FEATURE=...``.

@see [[tests/practical/AGENTS.md]]
@tested-by [[tests/scripts/test_practical_features.py]]
"""

import argparse
import fnmatch
import subprocess  # nosec B404 - fixed git argv, no shell
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

MAP = ROOT / "tests" / "practical" / "feature_map.yaml"
"""Path glob to feature id mapping."""


@dataclass(frozen=True)
class FeatureMap:
    """Parsed ``feature_map.yaml``."""

    smoke: tuple[str, ...]
    """Scenarios run for changes that touch every run (runner, orchestrator)."""

    paths: dict[str, tuple[str, ...]]
    """Glob to ids; ``smoke`` expanded, ``none`` empty."""


def load_map(path: Path) -> FeatureMap:
    """Read the feature map at ``path``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    smoke = tuple(data["smoke"])
    paths = {
        glob: smoke if ids == "smoke" else () if ids == "none" else tuple(ids)
        for glob, ids in data["paths"].items()
    }
    return FeatureMap(smoke, paths)


def _git(root: Path, *args: str) -> list[str]:
    result = subprocess.run(  # nosec B603 B607
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


def changed_paths(base: str, root: Path) -> list[str]:
    """Return paths changed since the merge base with ``base``, untracked included."""
    merge_base = _git(root, "merge-base", base, "HEAD")[0]
    changed = _git(root, "diff", "--name-only", merge_base)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    return sorted(set(changed + untracked))


def feature_ids(paths: Sequence[str], feature_map: FeatureMap) -> list[str]:
    """Return the sorted ids every glob matching one of ``paths`` names."""
    return sorted({
        feature
        for path in paths
        for glob, ids in feature_map.paths.items()
        if fnmatch.fnmatchcase(path, glob)
        for feature in ids
    })


def main(argv: Sequence[str] | None = None) -> int:
    """Print the comma-separated feature ids for the current branch."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--base", default="main")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    ids = feature_ids(changed_paths(args.base, args.root), load_map(MAP))
    print(",".join(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
