"""Static checks that keep the Sphinx sources consistent with the package.

Run ``uv run python docs/scripts/check_docs.py`` (no Sphinx build); prints
``path:line: message`` per finding and exits non-zero. It flags stale coalition-table
phrases, ``.cursor/skills`` paths and phase numbers in ``docs/``; ``:doc:`` targets
that do not exist; guide and theory pages outside the ``index.rst`` toctrees; public
modules without an ``automodule``; and repository links in ``docs/`` or
``CONTRIBUTING.md`` to missing files (the link checker skips them).
"""

import importlib
import pkgutil
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
EXAMPLES = ROOT / "examples"
EXCLUDED_DIRS = ("_build", "notebooks", "handbook", "_static")

STALE_PHRASES = (
    re.compile(r"2\^n\s*-\s*1"),
    re.compile(r"except the grand coalition", re.IGNORECASE),
)
CURSOR_SKILLS = re.compile(r"\.cursor/skills")
PHASE_NUMBER = re.compile(r"\b(?:Phase|Task)\s+\d+(?:\.\d+)?\b")
DOC_ROLE = re.compile(r":doc:`(?:[^`<]*<)?(?P<target>[^`>]+)>?`")
TOCTREE_ENTRY = re.compile(r"^\s{3}(?P<entry>[\w/.-]+)\s*$")
REPO_LINK = re.compile(
    r"https://github\.com/Pawlo77/nlp-shap/(?:blob|tree)/main/(?P<path>[^\s>`)?#]+)"
)
MARKDOWN_LINK = re.compile(r"\]\((?P<target>[^)\s#][^)\s]*)\)")
AUTOMODULE = re.compile(r"^\.\. automodule:: (?P<module>[\w.]+)", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    """One problem found in a documentation source."""

    path: Path
    line: int
    message: str

    def __str__(self) -> str:
        """Return ``path:line: message`` relative to the repository root."""
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def doc_sources(docs: Path = DOCS) -> list[Path]:
    """Return the hand-written ``.rst`` and ``.md`` sources under ``docs``."""
    return sorted(
        path
        for pattern in ("**/*.rst", "**/*.md")
        for path in docs.glob(pattern)
        if not any(part in EXCLUDED_DIRS for part in path.relative_to(docs).parts)
        and path.name not in {"AGENTS.md", "CLAUDE.md"}
    )


def _line_findings(path: Path, pattern: re.Pattern[str], message: str) -> list[Finding]:
    findings: list[Finding] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if pattern.search(line):
            findings.append(Finding(path, number, message))
    return findings


def check_phrases(paths: list[Path]) -> list[Finding]:
    """Flag stale coalition-table phrases, skills paths and phase numbers."""
    findings: list[Finding] = []
    for path in paths:
        for pattern in STALE_PHRASES:
            findings += _line_findings(
                path, pattern, f"stale phrase {pattern.pattern!r}"
            )
        findings += _line_findings(path, CURSOR_SKILLS, "use .agents/skills")
        if path.is_relative_to(DOCS) and path.name != "release_notes.rst":
            findings += _line_findings(path, PHASE_NUMBER, "internal phase/task number")
    return findings


def check_doc_links(
    paths: list[Path], docs: Path = DOCS, examples: Path = EXAMPLES
) -> list[Finding]:
    """Flag ``:doc:`` targets that do not exist.

    ``notebooks/<name>`` resolves to ``examples/<name>.ipynb``: the docs build copies
    the notebooks there, so the copy may be absent (fresh clone, CI).
    """
    findings: list[Finding] = []
    for path in paths:
        if path.suffix != ".rst":
            continue
        text = path.read_text(encoding="utf-8")
        for match in DOC_ROLE.finditer(text):
            target = match.group("target").strip()
            base = docs if target.startswith("/") else path.parent
            candidate = (base / target.lstrip("/")).resolve()
            copied = candidate.parent == (docs / "notebooks").resolve()
            if not any(
                candidate.with_suffix(suffix).is_file()
                or (candidate.parent / f"{candidate.name}{suffix}").is_file()
                for suffix in (".rst", ".md", ".ipynb")
            ) and not (copied and (examples / f"{candidate.name}.ipynb").is_file()):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(Finding(path, line, f":doc: target {target!r} missing"))
    return findings


def toctree_entries(index: Path) -> set[str]:
    """Return the entries of every toctree in ``index``."""
    entries: set[str] = set()
    in_toctree = False
    for line in index.read_text(encoding="utf-8").splitlines():
        if line.startswith(".. toctree::"):
            in_toctree = True
            continue
        if in_toctree and line and not line.startswith(" "):
            in_toctree = False
        if in_toctree:
            match = TOCTREE_ENTRY.match(line)
            if match and not line.strip().startswith(":"):
                entries.add(match.group("entry"))
    return entries


def check_toctree(docs: Path = DOCS) -> list[Finding]:
    """Flag guide and theory pages missing from the ``index.rst`` toctrees."""
    index = docs / "index.rst"
    entries = toctree_entries(index)
    findings: list[Finding] = []
    for folder in ("guides", "theory"):
        for page in sorted((docs / folder).glob("*.rst")):
            name = f"{folder}/{page.stem}"
            if name not in entries:
                findings.append(Finding(page, 1, f"{name} is not in a toctree"))
    return findings


def public_modules(package: str = "nlp_shap") -> list[str]:
    """Return every importable module of ``package`` without a private segment."""
    root = importlib.import_module(package)
    names = [package]
    names.extend(
        info.name
        for info in pkgutil.walk_packages(root.__path__, f"{package}.")
        if not any(part.startswith("_") for part in info.name.split("."))
    )
    return sorted(names)


def check_api_coverage(docs: Path = DOCS) -> list[Finding]:
    """Flag public modules without an ``automodule`` on an API page."""
    documented: set[str] = set()
    for page in [docs / "api.rst", *sorted((docs / "api").glob("*.rst"))]:
        documented |= set(AUTOMODULE.findall(page.read_text(encoding="utf-8")))
    return [
        Finding(docs / "api.rst", 1, f"module {name} has no automodule")
        for name in public_modules()
        if name not in documented
    ]


def check_repo_links(paths: list[Path]) -> list[Finding]:
    """Flag links to repository files that do not exist in the working tree."""
    findings: list[Finding] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        targets = [
            (match.start(), urllib.parse.unquote(match.group("path")))
            for match in REPO_LINK.finditer(text)
        ]
        if path.name == "CONTRIBUTING.md":
            targets += [
                (match.start(), match.group("target"))
                for match in MARKDOWN_LINK.finditer(text)
                if "://" not in match.group("target")
                and not match.group("target").startswith("mailto:")
            ]
        for start, target in targets:
            if not (ROOT / target).exists():
                line = text.count("\n", 0, start) + 1
                findings.append(
                    Finding(path, line, f"repository file {target!r} missing")
                )
    return findings


def run_checks(docs: Path = DOCS) -> list[Finding]:
    """Return every finding of every check."""
    sources = doc_sources(docs)
    readmes = [ROOT / "README.md", ROOT / "examples" / "README.md"]
    return (
        check_phrases(sources + [path for path in readmes if path.is_file()])
        + check_doc_links(sources, docs)
        + check_toctree(docs)
        + check_api_coverage(docs)
        + check_repo_links([*sources, ROOT / "CONTRIBUTING.md"])
    )


def main() -> int:
    """Print findings and return the exit code."""
    findings = run_checks()
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
