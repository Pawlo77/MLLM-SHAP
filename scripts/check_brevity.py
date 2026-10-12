"""Check the size budgets of rule ``brevity`` (``.cursor/rules/brevity.mdc``).

Usage::

    uv run python scripts/check_brevity.py          # exit 1 on any file over budget
    uv run python scripts/check_brevity.py --list   # every check, count and budget

Words exclude frontmatter, HTML comments and fenced or literal blocks. Files named in
``.agents/skills/VENDORED.md`` and ``.claude/agents/VENDORED.md`` are exempt.

<<<<<<< before updating
@domain tooling
=======
The budgets every repository shares are :data:`BASE_BUDGETS`; a repository adds its own
(``[[tool.agent-kit.brevity.budget]]``), grandfathered caps (``existing``), skipped
folders (``skip-dirs``, ``skip-paths``), release-note files (``bullet-files``) and the
folders of notebooks (``notebook-dirs``, default ``examples``) in
``[tool.agent-kit.brevity]``.
>>>>>>> after updating
"""

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

<<<<<<< before updating
=======
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import load as load_config

>>>>>>> after updating
ROOT = Path(__file__).resolve().parent.parent
"""Repository root."""

LONG_SENTENCE = 35
"""Words above which a sentence counts as long."""


@dataclass(frozen=True)
class Budget:
    """Limits for the files matching any of ``patterns``; the first match wins."""

    patterns: tuple[str, ...]
    words: int | None = None
    lines: int | None = None
    sentences: bool = False
    """Allow at most two sentences over ``LONG_SENTENCE`` words."""


<<<<<<< before updating
BUDGETS = (
    Budget(("AGENTS.md",), words=500, lines=80),
    Budget(("*/AGENTS.md",), lines=60),
=======
BASE_BUDGETS = (
    Budget(("AGENTS.md",), words=500, lines=80),
    Budget(("*/AGENTS.md",), lines=15),
>>>>>>> after updating
    Budget((".claude/rules/always.md",), words=700),
    Budget((".agents/skills/*/SKILL.md",), words=500, lines=100),
    Budget((".agents/skills/*.md",), words=400),
    Budget((".cursor/rules/*.mdc",), words=400, lines=60),
    Budget((".claude/agents/*.md",), words=400, lines=40),
    Budget((".cursor/commands/*.md",), lines=15),
<<<<<<< before updating
    Budget(("knowledge/issues/Fixed issues.md",), 800, sentences=True),
    Budget(("knowledge/decisions/*.md",), 700, sentences=True),
    Budget(("knowledge/issues/*.md",), 350, sentences=True),
    Budget(("knowledge/architecture/*.md", "knowledge/handbook/*.md"), 700, None, True),
    Budget(("knowledge/mllm-shap.md",), words=600, sentences=True),
    Budget(("knowledge/*.md", "docs/release_notes.rst"), sentences=True),
    Budget(("README.md",), words=900, sentences=True),
    Budget(("CONTRIBUTING.md",), words=1500, sentences=True),
    Budget(("examples/README.md",), words=400, sentences=True),
    Budget(("docs/*.rst", "docs/*.md"), words=1200, sentences=True),
)
"""Budgets of the rule, most specific first."""

SKIP_DIRS = {"node_modules", "__pycache__", "_build"}
SKIP_PATHS = {"docs/handbook", "docs/notebooks", "knowledge/generated"}
=======
    Budget(("README.md",), words=900, sentences=True),
    Budget(("CONTRIBUTING.md",), words=1500, sentences=True),
)
"""Budgets of the rule that every repository shares; a project's come first."""

SKIP_DIRS = {"node_modules", "__pycache__", "_build"}
>>>>>>> after updating
HIDDEN_KEPT = {".agents", ".claude", ".cursor"}
UNIT_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#|\.\.\s|[=\-~^\"'`*+]{3,}\s*$)")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
BULLET = re.compile(r"^(\s*)[-*]\s")
LITERAL_DIRECTIVE = re.compile(
    r"^\s*\.\. (code-block|code|sourcecode|math|testcode|testoutput|doctest|"
    r"jupyter-execute|ipython|literalinclude|toctree|autosummary|automodule)::"
)


@dataclass(frozen=True)
class Measure:
    """One count against one budget."""

    path: str
    count: int
    budget: int
    kind: str

    @property
    def over(self) -> bool:
        """Whether the count exceeds the budget."""
        return self.count > self.budget

    def __str__(self) -> str:
        """Render as ``path: count > budget (kind)``."""
        sign = ">" if self.over else "<="
        return f"{self.path}: {self.count} {sign} {self.budget} ({self.kind})"


<<<<<<< before updating
=======
@dataclass(frozen=True)
class Settings:
    """Per-repository additions, read from ``[tool.agent-kit.brevity]``."""

    budgets: tuple[Budget, ...] = BASE_BUDGETS
    existing: dict[tuple[str, str], int] | None = None
    skip_dirs: frozenset[str] = frozenset(SKIP_DIRS)
    skip_paths: frozenset[str] = frozenset()
    bullet_files: frozenset[str] = frozenset()
    notebook_dirs: tuple[str, ...] = ("examples/",)


def settings(root: Path = ROOT) -> Settings:
    """Return the budgets and skips of the repository at ``root``."""
    raw = load_config(root).get("brevity", {})
    table = raw if isinstance(raw, dict) else {}
    own = tuple(
        Budget(
            tuple(item["patterns"]),
            words=item.get("words"),
            lines=item.get("lines"),
            sentences=bool(item.get("sentences", False)),
        )
        for item in table.get("budget", [])
    )
    existing = {
        (key.rsplit("|", 1)[0], key.rsplit("|", 1)[1]): int(value)
        for key, value in table.get("existing", {}).items()
    }
    folders = table.get("notebook-dirs", ["examples"])
    return Settings(
        budgets=own + BASE_BUDGETS,
        existing=existing,
        skip_dirs=frozenset(SKIP_DIRS | set(table.get("skip-dirs", ()))),
        skip_paths=frozenset(table.get("skip-paths", ())),
        bullet_files=frozenset(table.get("bullet-files", ())),
        notebook_dirs=tuple(f"{folder.strip('/')}/" for folder in folders),
    )


>>>>>>> after updating
def frontmatter_lines(text: str) -> int:
    """Return the number of lines of a leading ``---`` frontmatter block."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return 0
    ends = [i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"]
    return ends[0] + 1 if ends else 0


def prose(text: str, suffix: str) -> str:
    """Blank frontmatter, comments and code, keeping line numbers."""
    text = re.sub(
        r"<!--.*?-->", lambda m: "\n" * m[0].count("\n"), text, flags=re.DOTALL
    )
    lines = text.splitlines()
    out = [""] * frontmatter_lines(text)
    fence = ""
    literal: int | None = None
    for line in lines[len(out) :]:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
<<<<<<< before updating
=======
        if stripped.startswith("$$"):  # display math is not prose
            out.append("")
            continue
>>>>>>> after updating
        # RST section underlines start with ``~~~``; fences exist only in Markdown.
        if fence or (suffix != ".rst" and stripped.startswith(("```", "~~~"))):
            if not fence:
                fence = stripped[:3]
            elif stripped.startswith(fence):
                fence = ""
            out.append("")
            continue
        if literal is not None and (not stripped or indent > literal):
            out.append("")
            continue
        literal = None
        if suffix == ".rst" and LITERAL_DIRECTIVE.match(line):
            literal = indent
            out.append("")
            continue
        if suffix == ".rst" and stripped.endswith("::"):
            literal = indent
        out.append(line)
    return "\n".join(out)


def words(text: str) -> int:
    """Count whitespace-separated tokens that contain a letter or digit."""
    return sum(1 for token in text.split() if re.search(r"\w", token))


def long_sentences(text: str) -> int:
    """Count sentences over ``LONG_SENTENCE`` words; items, cells and blocks split."""
    units: list[str] = []
    current: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.lstrip().startswith("|") or not line.strip() or UNIT_START.match(line):
            units.append(" ".join(current))
            current = []
        if line.lstrip().startswith("|"):
            units.extend(line.split("|"))
        elif line.strip():
            current.append(line)
    return sum(
        1
        for unit in units
        for sentence in SENTENCE_END.split(unit)
        if words(sentence) > LONG_SENTENCE
    )


def worst(items: list[Measure]) -> list[Measure]:
    """Return the items over budget, or the largest one when all pass."""
    over = [item for item in items if item.over]
    return over or sorted(items, key=lambda item: item.count)[-1:]


def bullets(rel: str, text: str) -> list[Measure]:
    """Measure each release-note bullet (continuation lines included)."""
    items: list[Measure] = []
    start, indent, count = 0, 0, 0
    for number, line in enumerate([*text.splitlines(), ""], 1):
        match = BULLET.match(line)
        continued = line.strip() and len(line) - len(line.lstrip()) > indent
        if start and (match or not continued):
            items.append(
                Measure(f"{rel}:{start}", count, 60, "release-note bullet words")
            )
            start = 0
        if match:
            start, indent, count = number, len(match[1]), words(line)
        elif start:
            count += words(line)
    return worst(items)


def notebook(rel: str, text: str) -> list[Measure]:
    """Measure markdown words and code lines per cell, and markdown words in total."""
    cells: list[Measure] = []
    total = 0
    for index, cell in enumerate(json.loads(text).get("cells", [])):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        where = f"{rel}[cell {index}]"
        if cell.get("cell_type") == "markdown":
            count = words(prose(source, ".md"))
            total += count
            cells.append(Measure(where, count, 150, "markdown cell words"))
        elif cell.get("cell_type") == "code":
            cells.append(
                Measure(where, len(source.splitlines()), 40, "code cell lines")
            )
    return [Measure(rel, total, 900, "notebook markdown words"), *worst(cells)]


def docstring(rel: str, text: str) -> list[Measure]:
    """Measure the module docstring's line span."""
    try:
        body = ast.parse(text).body
    except SyntaxError:
        return []
    first = body[0] if body else None
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)):
        return []
    span = (first.end_lineno or first.lineno) - first.lineno + 1
    return [Measure(rel, span, 12, "module docstring lines")]


<<<<<<< before updating
def by_budget(rel: str, text: str, budget: Budget) -> list[Measure]:
=======
def by_budget(
    rel: str, text: str, budget: Budget, bullet_files: frozenset[str] = frozenset()
) -> list[Measure]:
>>>>>>> after updating
    """Measure a text file against its budget."""
    body = prose(text, Path(rel).suffix)
    found: list[Measure] = []
    if budget.words is not None:
        found.append(Measure(rel, words(body), budget.words, "words"))
    if budget.lines is not None:
        count = len(text.splitlines()) - frontmatter_lines(text)
        found.append(Measure(rel, count, budget.lines, "lines"))
    if budget.sentences:
        found.append(Measure(rel, long_sentences(body), 2, "sentences over 35 words"))
<<<<<<< before updating
    if rel == "docs/release_notes.rst":
=======
    if rel in bullet_files:
>>>>>>> after updating
        found.extend(bullets(rel, body))
    return found


def vendored(root: Path) -> set[str]:
    """Return path prefixes of the vendored skills and agents."""
    prefixes: set[str] = set()
    for table, template in (
        (".agents/skills/VENDORED.md", ".agents/skills/{}/"),
        (".claude/agents/VENDORED.md", ".claude/agents/{}.md"),
    ):
        path = root / table
        rows = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        in_table = False
        for row in rows:
            if re.fullmatch(r"\|[\s:|-]+\|", row.strip()):
                in_table = True
            elif in_table and row.startswith("|"):
                names = row.split("|")[1].split("(")[0].split(",")
                prefixes.update(template.format(name.strip()) for name in names)
            else:
                in_table = False
    return prefixes


<<<<<<< before updating
def files(root: Path) -> Iterator[Path]:
    """Yield the repository files, skipping caches, builds and generated trees."""
=======
def files(root: Path, cfg: Settings | None = None) -> Iterator[Path]:
    """Yield the repository files, skipping caches, builds and generated trees."""
    cfg = cfg or settings(root)
>>>>>>> after updating
    for folder, dirs, names in root.walk():
        dirs[:] = [
            name
            for name in dirs
<<<<<<< before updating
            if name not in SKIP_DIRS
            and (not name.startswith(".") or name in HIDDEN_KEPT)
            and (folder / name).relative_to(root).as_posix() not in SKIP_PATHS
=======
            if name not in cfg.skip_dirs
            and (not name.startswith(".") or name in HIDDEN_KEPT)
            and (folder / name).relative_to(root).as_posix() not in cfg.skip_paths
>>>>>>> after updating
        ]
        yield from (folder / name for name in names if not (folder / name).is_symlink())


<<<<<<< before updating
def measure(root: Path) -> list[Measure]:
    """Return every measure for the files under ``root``."""
    exempt = tuple(vendored(root))
    found: list[Measure] = []
    for path in sorted(files(root)):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(exempt) or path.name == "VENDORED.md":
            continue
        if path.suffix == ".py" and not rel.startswith("examples/"):
            found.extend(docstring(rel, path.read_text(encoding="utf-8")))
        elif path.suffix == ".ipynb" and rel.startswith("examples/"):
            found.extend(notebook(rel, path.read_text(encoding="utf-8")))
        else:
            budget = next(
                (b for b in BUDGETS if any(fnmatch(rel, p) for p in b.patterns)), None
            )
            if budget is not None:
                found.extend(by_budget(rel, path.read_text(encoding="utf-8"), budget))
    return found
=======
def grandfathered(item: Measure, existing: dict[tuple[str, str], int]) -> Measure:
    """Return ``item`` with its ``existing`` cap as budget when it has one."""
    cap = existing.get((item.path, item.kind))
    if cap is None:
        return item
    return Measure(item.path, item.count, cap, f"{item.kind}, existing size")


def measure(root: Path) -> list[Measure]:
    """Return every measure for the files under ``root``."""
    cfg = settings(root)
    exempt = tuple(vendored(root))
    found: list[Measure] = []
    for path in sorted(files(root, cfg)):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(exempt) or path.name == "VENDORED.md":
            continue
        in_notebooks = rel.startswith(cfg.notebook_dirs)
        if path.suffix == ".py" and not in_notebooks:
            found.extend(docstring(rel, path.read_text(encoding="utf-8")))
        elif path.suffix == ".ipynb" and in_notebooks:
            found.extend(notebook(rel, path.read_text(encoding="utf-8")))
        else:
            budget = next(
                (b for b in cfg.budgets if any(fnmatch(rel, p) for p in b.patterns)),
                None,
            )
            text = (
                path.read_text(encoding="utf-8")
                if budget or rel in cfg.bullet_files
                else ""
            )
            if budget is not None:
                found.extend(by_budget(rel, text, budget, cfg.bullet_files))
            elif rel in cfg.bullet_files:
                found.extend(bullets(rel, prose(text, path.suffix)))
    return [grandfathered(item, cfg.existing or {}) for item in found]
>>>>>>> after updating


def main(argv: list[str] | None = None) -> int:
    """Print the measures over budget (all of them with ``--list``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print every measure")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    found = measure(args.root.resolve())
    over = [item for item in found if item.over]
    for item in found if args.list else over:
        print(item)
    if over:
        print(f"check_brevity: {len(over)} measure(s) over budget")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
