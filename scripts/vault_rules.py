"""Vault rules of mllm-shap on top of the shared checker ``check_vault.py``.

Checks the note kinds (decision log, paper list, concept), source pointers, the
abbreviation glossary, plan headers, code traceability tags and architecture-note
freshness. The note types, statuses and sections are the schema in pyproject.toml.

@domain tooling
"""

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from check_vault import (
    LINK,
    Finding,
    comments_and_docstrings,
    frontmatter,
    git,
    has_frontmatter,
    is_placeholder,
    line_of,
    module_docstring,
)

ENTRIES = {
    "decision-log": (
        re.compile(r"(\d{4}-\d{2}-\d{2}) — \S.*"),
        ("Decision:", "Why:", "Consequences:"),
    ),
    "paper-list": (
        re.compile(r"\S+ \d{4} — \S.*"),
        ("Citation:", "Claim:", "Used by:"),
    ),
}
"""Per list type: the ``## `` heading pattern of an entry and the labels that must
start a line inside it. Decision-log entries are dated and listed newest first."""

PLAN_HEADER = ("**Goal:**", "**Architecture:**", "**Tech Stack:**")
"""Writing-plans header lines a ``type: plan`` file must contain."""

POINTERS = re.compile(
    r"\bpp?\. ?\d|\bTables? \d|\bFig\.? \d|\bFigure \d|\bSec\.? \d|\bSection \d"
    r"|[Ff]ootnote \d|\bAppendix [A-Z\d]|\bEqs?\. ?\(?\d|\bProposition \d"
    r"|\bLemma \d|\bTheorem \d"
)
"""Source pointers (page, table, figure, equation numbers) banned in the papers list."""

TAG_DOMAIN = re.compile(r"@domain\s+\S+")
TAG_SEE = re.compile(r"@see\s+\[\[([^\]|#]+)")
PACKAGE = Path("src") / "mllm_shap"


def mask_prose(text: str) -> str:
    """Blank frontmatter, code, math, headings, links and tags for prose checks."""
    chars = list(text)

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "

    head = re.match(r"---\n.*?\n---\n", text, re.DOTALL)
    if head:
        blank(0, head.end())
    for pattern, flags in ((r"```.*?```", re.DOTALL), (r"\$\$.*?\$\$", re.DOTALL)):
        for match in re.finditer(pattern, text, flags):
            blank(match.start(), match.end())
    for pattern in (
        r"(?<!\\)\$[^$\n]+?(?<!\\)\$",
        r"`[^`\n]*`",
        r"\[\[[^\]]*\]\]",
        r"\]\([^)]*\)",
        r"(?m)^#.*$",
        r"<[^>\n]+>",
    ):
        for match in re.finditer(pattern, text):
            blank(match.start(), match.end())
    return "".join(chars)


def load_glossary(root: Path) -> dict[str, dict[str, Any]]:
    """Return the abbreviation glossary (empty when the file is missing)."""
    path = root / "scripts" / "glossary.yaml"
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(key): value for key, value in loaded.items()}


def unexpanded(body: str, glossary: dict[str, dict[str, Any]]) -> list[str]:
    """Return glossary abbreviations used in ``body`` without their expansion."""
    lowered = body.lower()
    missing: list[str] = []
    for abbreviation, spec in glossary.items():
        pattern = r"(?<![\w&])" + re.escape(abbreviation) + r"s?(?![\w&])"
        if not re.search(pattern, body):
            continue
        forms = [spec["expansion"], spec.get("plural") or "", *spec.get("also", [])]
        if not any(form and form.lower() in lowered for form in forms):
            missing.append(f"{abbreviation} ({spec['expansion']})")
    return missing


def check_entries(rel: str, text: str, kind: str) -> list[Finding]:
    """Return findings for the ``## `` entries of a decision log or papers list."""
    pattern, labels = ENTRIES[kind]
    findings: list[Finding] = []
    dates: list[str] = []
    parts = re.split(r"(?m)^## (.+?)[ \t]*$", text)
    offset = len(parts[0])
    for heading, body in zip(parts[1::2], parts[2::2], strict=True):
        line = line_of(text, offset)
        offset += len(heading) + len(body) + 3
        match = pattern.fullmatch(heading)
        if match is None:
            findings.append(Finding(rel, line, f"entry heading '{heading}' malformed"))
            continue
        if match.groups():
            dates.append(match.group(1))
        findings += [
            Finding(rel, line, f"entry '{heading}' has no '{label}' line")
            for label in labels
            if not re.search(rf"(?m)^{re.escape(label)} ", body)
        ]
    if dates != sorted(dates, reverse=True):
        findings.append(Finding(rel, 1, "entries are not listed newest first"))
    return findings


def check_note(vault: Any, path: Path, text: str) -> list[Finding]:
    """Return the kind-specific findings of one knowledge note."""
    try:
        meta = frontmatter(text)
    except yaml.YAMLError:
        return []
    kind = meta.get("type")
    if not has_frontmatter(text) or kind not in vault.cfg["schema"]["types"]:
        return []
    rel = vault.rel(path)
    findings: list[Finding] = []
    if kind == "concept":
        body = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        before = re.split(r"(?m)^## ", body, maxsplit=1)[0]
        prose = [
            line
            for line in before.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not prose:
            findings.append(Finding(rel, 1, "concept note needs a first paragraph"))
    if kind in ENTRIES:
        findings += check_entries(rel, text, str(kind))
    if kind == "generated":
        return findings
    body = mask_prose(text)
    if kind == "paper-list":
        findings += [
            Finding(rel, line_of(text, match.start()), f"source pointer '{match[0]}'")
            for match in POINTERS.finditer(body)
        ]
    glossary = load_glossary(vault.root)
    findings += [
        Finding(rel, 1, f"abbreviation {item} used but never spelled out")
        for item in unexpanded(body, glossary)
    ]
    return findings


def check_plan(vault: Any, path: Path, text: str) -> list[Finding]:
    """Return findings for one ``plans/*.md`` file's frontmatter and header."""
    if path.name.lower() == "readme.md":
        return []
    rel = vault.rel(path)
    if not has_frontmatter(text):
        return [
            Finding(
                rel, 1, "plan file has no frontmatter (type, status)", True, "strict"
            )
        ]
    try:
        meta = frontmatter(text)
    except yaml.YAMLError:
        return [Finding(rel, 1, "frontmatter is not valid YAML")]
    schema = vault.cfg["schema"]
    findings: list[Finding] = []
    if meta.get("type") not in schema["types"]:
        findings.append(Finding(rel, 1, f"type '{meta.get('type')}' not allowed"))
    if meta.get("type") == "plan":
        if meta.get("status") not in schema["statuses"]["plan"]:
            findings.append(
                Finding(rel, 1, f"plan status '{meta.get('status')}' not valid")
            )
        findings += [
            Finding(rel, 1, f"plan header line '{line}' missing")
            for line in PLAN_HEADER
            if line not in text
        ]
    return findings


def check_file(vault: Any, path: Path, text: str) -> list[Finding]:
    """Return the project findings for one note or plan."""
    if vault.under(path, "plans"):
        return check_plan(vault, path, text)
    return check_note(vault, path, text)


def code_files(root: Path) -> list[Path]:
    """Return package and test Python files checked for links and tags."""
    return sorted((root / PACKAGE).rglob("*.py")) + sorted(
        (root / "tests").rglob("*.py")
    )


def check_code(vault: Any, files: list[Path]) -> list[Finding]:
    """Return broken-link and traceability-tag findings for Python files."""
    findings: list[Finding] = []
    package = vault.root / PACKAGE
    for path in files:
        rel = vault.rel(path)
        text = comments_and_docstrings(path)
        for match in LINK.finditer(text):
            if is_placeholder(match.group(1)):
                continue
            problem = vault.link_problem(match)
            if problem:
                findings.append(Finding(rel, line_of(text, match.start()), problem))
        docstring = module_docstring(path)
        in_package = path.is_relative_to(package)
        if not (in_package or path.name.startswith("test_")):
            continue
        if in_package and not TAG_DOMAIN.search(docstring):
            findings.append(Finding(rel, 1, "missing @domain tag", True, "tags"))
        see = TAG_SEE.search(docstring)
        if see is None:
            findings.append(Finding(rel, 1, "missing @see [[...]] tag", True, "tags"))
        elif in_package:
            target = vault.resolve(see.group(1))
            if target is None or target.suffix != ".md":
                findings.append(
                    Finding(
                        rel,
                        1,
                        f"@see target is not a note: {see.group(1)}",
                        True,
                        "tags",
                    )
                )
    return findings


def architecture_note_for(root: Path, changed: str) -> Path | None:
    """Return the architecture note path for a changed ``src/mllm_shap`` file."""
    parts = Path(changed).parts
    if len(parts) < 3 or parts[:2] != PACKAGE.parts:
        return None
    package = parts[2] if len(parts) > 3 else "core"
    return root / "knowledge" / "architecture" / f"{package}.md"


def check_freshness(vault: Any) -> list[Finding]:
    """Return warnings for architecture notes older than their package's changes."""
    base = git(vault.root, "merge-base", "HEAD", "origin/main")
    if not base:
        return []
    changed = git(vault.root, "diff", "--name-only", f"{base}...HEAD") or ""
    notes: dict[Path, str] = {}
    for name in changed.splitlines():
        note = architecture_note_for(vault.root, name)
        if note is None:
            continue
        package_dir = str(Path(name).parent) if note.stem != "core" else name
        newest = git(
            vault.root, "log", "-1", "--format=%cs", f"{base}..HEAD", "--", package_dir
        )
        if newest and newest > notes.get(note, ""):
            notes[note] = newest
    findings: list[Finding] = []
    for note, newest in sorted(notes.items()):
        rel = note.relative_to(vault.root).as_posix()
        if not note.is_file():
            findings.append(
                Finding(rel, 1, "architecture note missing", True, "strict")
            )
            continue
        meta = frontmatter(note.read_text(encoding="utf-8"))
        stamp = str(meta.get("updated") or meta.get("created") or "")
        if isinstance(meta.get("updated"), date):
            stamp = meta["updated"].isoformat()
        if stamp < newest:
            message = (
                f"'updated' {stamp or 'missing'} is older than package change "
                f"{newest}; update the note"
            )
            findings.append(Finding(rel, 1, message, True, "strict"))
    return findings


def check_vault(vault: Any) -> list[Finding]:
    """Return the whole-vault findings: code tags and note freshness."""
    return check_code(vault, code_files(vault.root)) + check_freshness(vault)
