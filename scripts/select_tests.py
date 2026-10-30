"""Run only the tests a branch can affect (the pre-push hook; CI runs them all).

Usage: ``uv run python scripts/select_tests.py [--base REF] [--list] [-- PYTEST ARGS]``

The changed files are the diff of ``HEAD`` against the merge base with ``--base``
(default: the upstream branch, else ``origin/main``). A test is selected when its import
closure reaches a changed Python file, or when it or one of its local modules names a
changed file (path, name or dotted module in a string: ``python -m`` calls, data files).
Docs, notes and agent settings select nothing. The whole suite runs for a change to the
test setup, the dependencies or the build files, a deleted Python file, or when no base
is found; ``[tool.agent-kit.tests]`` ``full`` adds glob patterns that force it.
"""

import argparse
import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import load as load_config

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts", "tests")
"""Where imports resolve; ``src`` and ``scripts`` are import roots as well."""
FULL = (
    "pyproject.toml", "uv.lock", "Makefile", "kit.mk", ".python-version",
    "conftest.py", "*/conftest.py", "pytest.ini", "tox.ini", "setup.cfg",
)  # fmt: skip
"""Changes that can move any test."""
IGNORED = (
    "*.md", "*.rst", "LICENSE", "*.cff", "knowledge/*", "plans/*", "research/*",
    "report/*", "docs/*", ".remember/*", ".obsidian/*", ".github/*", ".cursor/*",
    ".claude/*", ".codex/*", ".agents/*", ".gitignore", ".gitattributes",
    ".pre-commit-config.yaml", ".editorconfig", "*.code-workspace",
)  # fmt: skip
"""Changes no test reads by name (a test that does is caught by its own mention)."""


def _git(root: Path, *args: str) -> str | None:
    """Return the output of a git command, or ``None`` when it fails."""
    done = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return done.stdout.strip() if done.returncode == 0 else None


def changed_files(root: Path, base: str | None) -> tuple[list[str], list[str]] | None:
    """Return (existing, deleted) paths changed since the merge base, or ``None``."""
    for ref in (base, "@{upstream}", "origin/main", "origin/master"):
        point = ref and _git(root, "merge-base", "HEAD", ref)
        if point:
            break
    else:
        return None
    names = _git(root, "diff", "--name-only", point, "HEAD") or ""
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all") or ""
    paths = set(names.splitlines()) | {line[3:] for line in dirty.splitlines()}
    present = sorted(p for p in paths if (root / p).is_file())
    return present, sorted(paths - set(present))


def _modules(root: Path) -> dict[str, Path]:
    """Dotted module name -> file, for each way a name can be imported."""
    found: dict[str, Path] = {}
    for folder in CODE_DIRS:
        for path in sorted((root / folder).rglob("*.py")):
            rel = path.relative_to(root)
            for start in {root, root / "src", root / "scripts"}:
                if path.is_relative_to(start):
                    parts = list(path.relative_to(start).with_suffix("").parts)
                    if parts[-1] == "__init__":
                        parts.pop()
                    if parts:
                        found.setdefault(".".join(parts), rel)
    return found


def _from_targets(
    node: ast.ImportFrom, package: list[str], modules: dict[str, Path]
) -> list[str]:
    """Module names a ``from x import y`` reaches: ``x.y`` if a module, else ``x``."""
    up = package[: len(package) - node.level + 1] if node.level else []
    base = ".".join([*up, *([node.module] if node.module else [])])
    return [
        f"{base}.{a.name}" if f"{base}.{a.name}" in modules else base
        for a in node.names
    ]


def _refs(tree: ast.AST, here: Path, modules: dict[str, Path]) -> tuple[set, set, set]:
    """Return (imported files, parent packages, string constants) of one parsed file.

    A parent package is run by the import but only a change to it matters, so it is a
    leaf; following its re-exports would tie every test to every module of the package.
    """
    strong: set[Path] = set()
    leaves: set[Path] = set()
    strings: set[str] = set()
    siblings = {p.stem: p for p in here.parent.glob("*.py")}
    own = next((m for m, p in modules.items() if p == here), "")
    package = own.split(".") if here.name == "__init__.py" else own.split(".")[:-1]

    def add(name: str) -> None:
        if name in modules:
            strong.add(modules[name])
        elif name in siblings:  # a test folder without __init__ is on sys.path
            strong.add(siblings[name])
        while name := name.rpartition(".")[0]:
            if name in modules:
                leaves.add(modules[name])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for name in _from_targets(node, package, modules):
                add(name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
            add(node.value[:200].removesuffix(".py"))
    return strong, leaves, strings


def graph(root: Path) -> tuple[dict[Path, tuple[set, set]], dict[Path, str]]:
    """Return the (imports, parent packages) and string constants of every code file."""
    modules = _modules(root)
    known = set(modules.values())
    edges: dict[Path, tuple[set, set]] = {}
    texts: dict[Path, str] = {}
    for rel in known:
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        strong, leaves, strings = _refs(tree, rel, modules)
        # a test also runs the conftest.py of every folder above it
        above = {rel.parents[i] / "conftest.py" for i in range(len(rel.parents) - 1)}
        edges[rel] = (strong | (above & known), leaves)
        texts[rel] = "\n".join(strings)
    return edges, texts


def _closure(start: Path, edges: dict[Path, tuple[set, set]]) -> set[Path]:
    seen, todo = {start}, [start]
    while todo:
        strong, leaves = edges.get(todo.pop(), (set(), set()))
        seen |= leaves
        for nxt in strong - seen:
            seen.add(nxt)
            todo.append(nxt)
    return seen


def select(root: Path, changed: list[str]) -> list[str]:
    """Return the test files affected by ``changed`` (existing paths)."""
    edges, texts = graph(root)
    hit = {Path(p) for p in changed}
    names = {p: (Path(p).name, p) for p in changed}
    picked: list[str] = []
    for test in sorted(p for p in edges if p.parts[0] == "tests"):
        if not (test.name.startswith("test_") or test.name.endswith("_test.py")):
            continue
        reach = _closure(test, edges)
        mentioned = any(
            any(token in texts.get(file, "") for token in names[p])
            for file in reach
            for p in names
        )
        if reach & hit or mentioned:
            picked.append(test.as_posix())
    return picked


def plan(root: Path, base: str | None) -> list[str] | None:
    """Return the tests to run, or ``None`` for the whole suite."""
    found = changed_files(root, base)
    if found is None:
        return None
    present, deleted = found
    extra = tuple(load_config(root).get("tests", {}).get("full", ()))
    if any(p.endswith(".py") for p in deleted):
        return None
    if any(fnmatch.fnmatch(p, pat) for p in present for pat in (*FULL, *extra)):
        return None
    relevant = [p for p in present if not any(fnmatch.fnmatch(p, g) for g in IGNORED)]
    return select(root, relevant) if relevant else []


def main(argv: list[str] | None = None) -> int:
    """Print or run the selection; return the exit status."""
    args = list(sys.argv[1:] if argv is None else argv)
    extra = args[args.index("--") + 1 :] if "--" in args else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", help="ref to diff against (default: upstream)")
    parser.add_argument("--list", action="store_true", help="print, do not run")
    opts = parser.parse_args(args[: args.index("--")] if "--" in args else args)
    tests = plan(ROOT, opts.base)
    if tests is None:
        print("select_tests: running the whole suite", file=sys.stderr)
        command = ["pytest", *extra]
    elif not tests:
        print("select_tests: no test is affected by the changes", file=sys.stderr)
        return 0
    else:
        print(f"select_tests: {len(tests)} affected test files", file=sys.stderr)
        command = ["pytest", *tests, *extra]
    if opts.list:
        print("\n".join(tests or ["(all)"]))
        return 0
    return subprocess.run(["uv", "run", *command], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
