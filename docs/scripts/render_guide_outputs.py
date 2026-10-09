"""Execute guide snippets and write their output for the ``guide-result`` directive.

A ``.. code-block:: python`` in ``docs/guides`` or ``docs/theory`` followed by
``.. guide-result:: <id>`` runs; snippets of a page share one namespace, in order,
inside a temporary directory. Output goes to ``_static/guide_outputs/<id>.txt``;
``:figure:`` directives get the opened figures as ``<id>.png``.

Usage::

    uv run python docs/scripts/render_guide_outputs.py            # render all
    uv run python docs/scripts/render_guide_outputs.py --check    # list only
"""

import argparse
import contextlib
import io
import os
import re
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUTPUT_DIR = DOCS / "_static" / "guide_outputs"
PAGE_GLOBS = ("guides/*.rst", "theory/*.rst")

_CODE_BLOCK = re.compile(r"^(?P<indent>[ \t]*)\.\. code-block:: python\s*$")
_GUIDE_RESULT = re.compile(r"^(?P<indent>[ \t]*)\.\. guide-result:: (?P<id>\S+)\s*$")
_OPTION = re.compile(r"^[ \t]+:(?P<name>[\w-]+):")


@dataclass(frozen=True)
class GuideResult:
    """One ``guide-result`` directive attached to a snippet."""

    snippet_id: str
    figure: bool
    line: int


@dataclass(frozen=True)
class Snippet:
    """A runnable Python code block and the directives that show its output."""

    page: Path
    line: int
    code: str
    results: tuple[GuideResult, ...]

    @property
    def label(self) -> str:
        """Return ``page:line`` for error messages."""
        return f"{self.page.relative_to(ROOT)}:{self.line}"


@dataclass
class PageScan:
    """Snippets and stray directives found on one page."""

    page: Path
    snippets: list[Snippet] = field(default_factory=list)
    orphans: list[GuideResult] = field(default_factory=list)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_body(lines: list[str], start: int, indent: int) -> tuple[str, int]:
    """Return the dedented body of the directive opened at ``start - 1``."""
    index = start
    while index < len(lines) and _OPTION.match(lines[index]):
        index += 1
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and _indent_of(line) <= indent:
            break
        body.append(line)
        index += 1
    while body and not body[-1].strip():
        body.pop()
    return textwrap.dedent("\n".join(body)).strip("\n") + "\n", index


def _directive_options(lines: list[str], start: int) -> tuple[set[str], int]:
    names: set[str] = set()
    index = start
    while index < len(lines):
        match = _OPTION.match(lines[index])
        if match is None:
            break
        names.add(match.group("name"))
        index += 1
    return names, index


def _skip_blank(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def scan_page(page: Path) -> PageScan:
    """Return the runnable snippets and orphan ``guide-result`` directives."""
    lines = page.read_text(encoding="utf-8").splitlines()
    scan = PageScan(page=page)
    attached: set[int] = set()
    index = 0
    while index < len(lines):
        match = _CODE_BLOCK.match(lines[index])
        if match is None:
            index += 1
            continue
        indent = len(match.group("indent"))
        code, after = _block_body(lines, index + 1, indent)
        results: list[GuideResult] = []
        cursor = _skip_blank(lines, after)
        while cursor < len(lines):
            result = _GUIDE_RESULT.match(lines[cursor])
            if result is None or len(result.group("indent")) != indent:
                break
            attached.add(cursor)
            options, end = _directive_options(lines, cursor + 1)
            results.append(
                GuideResult(result.group("id"), "figure" in options, cursor + 1)
            )
            cursor = _skip_blank(lines, end)
        if results:
            scan.snippets.append(Snippet(page, index + 1, code, tuple(results)))
        index = after
    for number, line in enumerate(lines):
        result = _GUIDE_RESULT.match(line)
        if result is None or number in attached:
            continue
        options, _ = _directive_options(lines, number + 1)
        scan.orphans.append(
            GuideResult(result.group("id"), "figure" in options, number + 1)
        )
    return scan


def scan_docs(docs: Path = DOCS) -> list[PageScan]:
    """Scan every guide and theory page in document order."""
    pages = sorted(path for pattern in PAGE_GLOBS for path in docs.glob(pattern))
    return [scan_page(page) for page in pages]


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _new_figures(before: set[int]) -> list[object]:
    if "matplotlib.pyplot" not in sys.modules:
        return []
    import matplotlib.pyplot as plt

    return [plt.figure(number) for number in plt.get_fignums() if number not in before]


def _open_figure_numbers() -> set[int]:
    if "matplotlib.pyplot" not in sys.modules:
        return set()
    import matplotlib.pyplot as plt

    return set(plt.get_fignums())


def run_snippet(snippet: Snippet, namespace: dict[str, object]) -> tuple[str, list]:
    """Execute one snippet; return its stdout and the figures it opened."""
    before = _open_figure_numbers()
    buffer = io.StringIO()
    compiled = compile(snippet.code, snippet.label, "exec")
    try:
        with contextlib.redirect_stdout(buffer):
            exec(compiled, namespace)  # nosec B102 - trusted docs code
    except Exception as exc:
        msg = f"guide snippet {snippet.label} failed: {exc!r}"
        raise RuntimeError(msg) from exc
    return buffer.getvalue().rstrip().lstrip("\n"), _new_figures(before)


def _write_outputs(
    snippet: Snippet, text: str, figures: list, output_dir: Path
) -> list[Path]:
    written: list[Path] = []
    first = snippet.results[0]
    if text:
        path = output_dir / f"{first.snippet_id}.txt"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        written.append(path)
    figure_results = [result for result in snippet.results if result.figure]
    if len(figures) < len(figure_results):
        msg = (
            f"guide snippet {snippet.label} opened {len(figures)} figure(s) but "
            f"{len(figure_results)} guide-result directive(s) expect one"
        )
        raise RuntimeError(msg)
    for result, figure in zip(figure_results, figures, strict=False):
        path = output_dir / f"{result.snippet_id}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        written.append(path)
    if not first.figure and not text:
        msg = f"guide snippet {snippet.label} printed nothing for {first.snippet_id}"
        raise RuntimeError(msg)
    for result in snippet.results[1:]:
        if not result.figure:
            msg = (
                f"guide snippet {snippet.label}: only the first guide-result shows "
                f"text; {result.snippet_id!r} needs the :figure: option"
            )
            raise RuntimeError(msg)
    if "matplotlib.pyplot" in sys.modules:
        import matplotlib.pyplot as plt

        for figure in figures:
            plt.close(figure)
    return written


def render(docs: Path = DOCS, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Run every snippet and write outputs; remove outputs no directive uses."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for scan in scan_docs(docs):
        if scan.orphans:
            orphan = scan.orphans[0]
            msg = (
                f"{scan.page.relative_to(ROOT)}:{orphan.line}: guide-result "
                f"{orphan.snippet_id!r} does not follow a python code-block"
            )
            raise RuntimeError(msg)
        if not scan.snippets:
            continue
        namespace: dict[str, object] = {"__name__": "__guide__"}
        with tempfile.TemporaryDirectory() as tmp, _working_directory(Path(tmp)):
            for snippet in scan.snippets:
                text, figures = run_snippet(snippet, namespace)
                written.extend(_write_outputs(snippet, text, figures, output_dir))
                print(f"wrote {', '.join(r.snippet_id for r in snippet.results)}")
    keep = {path.name for path in written}
    for stale in output_dir.iterdir():
        if stale.is_file() and stale.name not in keep:
            stale.unlink()
            print(f"removed stale {stale.name}")
    return written


def main() -> None:
    """Render guide outputs, or list snippets with ``--check``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="list snippets without running them"
    )
    args = parser.parse_args()
    if args.check:
        for scan in scan_docs():
            for snippet in scan.snippets:
                ids = ", ".join(result.snippet_id for result in snippet.results)
                print(f"{snippet.label}: {ids}")
        return
    render()


if __name__ == "__main__":
    main()
