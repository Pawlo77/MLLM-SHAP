"""Report template-managed files that were edited in this repository.

Usage:
    python scripts/template_drift.py [--copier "uvx --from copier==9.18.2 copier"]

Renders the template at the commit recorded in ``.copier-answers.yml`` into a temporary
directory (tasks skipped) and compares every rendered file with the repository's copy.
Exit 0: no drift; exit 1: lists the files that differ or are missing; exit 2: the render
failed. Edit the template instead, or move the change to a project-owned file.
Stdlib only.
"""

import argparse
import shlex
import subprocess  # nosec B404 - argv from the committed answers file, no shell
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ".copier-answers.yml"
SKIP = {ANSWERS}
"""Rendered files that are bookkeeping, not content."""

Renderer = Callable[[dict[str, str], Path], None]


def parse_answers(text: str) -> dict[str, str]:
    """Return the flat ``key: value`` pairs of a Copier answers file."""
    answers = {}
    for line in text.splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        answers[key.strip()] = value.strip().strip("'\"")
    return answers


def make_renderer(copier: list[str]) -> Renderer:
    """Return a renderer that runs ``copier copy`` at the recorded commit."""

    def render(answers: dict[str, str], dest: Path) -> None:
        data = [
            f"--data={key}={value}" for key, value in answers.items() if key[0] != "_"
        ]
        cmd = [*copier, "copy", "--trust", "--skip-tasks", "--defaults", "--quiet"]
        cmd += [
            f"--vcs-ref={answers['_commit']}",
            *data,
            answers["_src_path"],
            str(dest),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)  # nosec B603

    return render


def drift(root: Path, rendered: Path) -> list[str]:
    """Return repository paths whose content differs from the rendered template."""
    found = []
    for path in sorted(rendered.rglob("*")):
        rel = path.relative_to(rendered).as_posix()
        if not path.is_file() or rel in SKIP or ".git/" in f"{rel}/":
            continue
        mine = root / rel
        if not mine.is_file():
            found.append(f"{rel} (missing)")
        elif mine.read_bytes() != path.read_bytes():
            found.append(rel)
    return found


def main(argv: list[str] | None = None, render: Renderer | None = None) -> int:
    """Print the drifted files; see the module docstring for the exit codes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--copier", default="uvx --from copier==9.18.2 copier")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    answers_file = args.root / ANSWERS
    if not answers_file.is_file():
        print(f"template_drift: no {ANSWERS}; this repository is not from the template")
        return 2
    answers = parse_answers(answers_file.read_text(encoding="utf-8"))
    render = render or make_renderer(shlex.split(args.copier))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            render(answers, Path(tmp))
        except (subprocess.CalledProcessError, OSError, KeyError) as error:
            detail = getattr(error, "stderr", "") or error
            print(f"template_drift: render failed: {detail}", file=sys.stderr)
            return 2
        files = drift(args.root, Path(tmp))
    for name in files:
        print(f"drift: {name}")
    return 1 if files else 0


if __name__ == "__main__":
    raise SystemExit(main())
