"""Consistency checks for the mllm-shap Obsidian vault (repository root).

Usage:
    python scripts/check_vault.py [--quiet] [--strict] [--strict-tags] [--root DIR]

Prints ``path:line: message`` per finding and exits 1 on any error. Checks note
frontmatter, required sections, links, isolated notes, abbreviations, secrets, code
traceability tags and plan headers. ``--strict`` promotes architecture-note freshness
and plan warnings (CI); ``--strict-tags`` promotes missing ``@domain``/``@see`` tags.

@domain tooling
"""

import argparse
import io
import re
import subprocess  # nosec B404
import sys
import tokenize
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "hooks"))

from check_memory_files import maintainer_emails, scan_text  # noqa: E402

NOTE_TYPES = {
    "map",
    "dashboard",
    "module",
    "decision-log",
    "issue",
    "concept",
    "paper-list",
    "handbook",
    "solution",
    "plan",
    "generated",
}
"""Allowed values of the ``type`` frontmatter field."""

STATUSES = {
    "issue": {"open", "fixed", "wontfix", "deferred"},
    "plan": {"draft", "ready", "in progress", "done", "superseded"},
}
"""Valid ``status`` values per note type (required for these types)."""

REQUIRED_SECTIONS = {
    "module": ["Responsibility", "Public surface", "Data flow", "Invariants", "Tests"],
    "issue": ["Symptom", "Cause", "Repro", "Fix"],
    "solution": ["Problem", "What was tried", "What worked", "Reusable rule"],
}
"""``## `` headings each note type must contain."""

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

FIXED_ISSUES = "Fixed issues"
"""Issue note holding one table row per fixed bug; exempt from the issue sections."""

SKIPPED_FOLDERS = {"_templates"}
"""Folders under ``knowledge/`` that hold no vault notes: Templater templates."""

AGENT_FILES = {"AGENTS.md", "CLAUDE.md"}
"""Agent instructions beside the notes (``CLAUDE.md`` links to ``AGENTS.md``)."""

REQUIRED_FIELDS = ("title", "type", "domain", "created", "parent")
"""Frontmatter keys every knowledge note needs (``created`` optional when generated,
``parent`` absent on the map)."""

ROOTLESS_TYPES = {"map"}
"""Note types without a ``parent`` link."""

PLAN_HEADER = ("**Goal:**", "**Architecture:**", "**Tech Stack:**")
"""Writing-plans header lines a ``type: plan`` file must contain."""

POINTERS = re.compile(
    r"\bpp?\. ?\d|\bTables? \d|\bFig\.? \d|\bFigure \d|\bSec\.? \d|\bSection \d"
    r"|[Ff]ootnote \d|\bAppendix [A-Z\d]|\bEqs?\. ?\(?\d|\bProposition \d"
    r"|\bLemma \d|\bTheorem \d"
)
"""Source pointers (page, table, figure, equation numbers) banned in the papers list."""

LINK = re.compile(r"\[\[([^\]|#\\]+)(?:#([^\]|\\]*))?(?:\\?\|[^\]]*)?\]\]")
MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
"""Markdown link target; Obsidian resolves relative ones like wikilinks."""
"""Wikilink target and optional heading (``\\|`` is the escaped alias pipe used inside
Markdown tables)."""

FILE_EXTENSIONS = (
    ".py",
    ".md",
    ".pdf",
    ".ipynb",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".rst",
    ".txt",
    ".cfg",
    ".sh",
)
"""Link targets with these suffixes are vault paths to files."""

TAG_DOMAIN = re.compile(r"@domain\s+\S+")
TAG_SEE = re.compile(r"@see\s+\[\[([^\]|#]+)")


@dataclass(frozen=True)
class Finding:
    """One checker result."""

    path: str
    line: int
    message: str
    warning: bool = False

    def __str__(self) -> str:
        """Format the issue as ``path:line: message``."""
        level = "warning: " if self.warning else ""
        return f"{self.path}:{self.line}: {level}{self.message}"


def _notes_in(folder: Path) -> list[Path]:
    """Return the Markdown notes directly in ``folder``, agent files excluded."""
    return sorted(p for p in folder.glob("*.md") if p.name not in AGENT_FILES)


@dataclass
class Vault:
    """Notes and link targets of a vault rooted at ``root``."""

    root: Path
    notes: list[Path] = field(default_factory=list)
    plans: list[Path] = field(default_factory=list)
    memory: list[Path] = field(default_factory=list)
    by_stem: dict[str, list[Path]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "Vault":
        """Collect knowledge notes, plans and memory notes under ``root``."""
        vault = cls(root=root)
        vault.notes = sorted(
            path
            for path in (root / "knowledge").rglob("*.md")
            if not SKIPPED_FOLDERS & set(path.relative_to(root).parts)
            and path.name not in AGENT_FILES
        )
        vault.plans = _notes_in(root / "plans")
        vault.memory = _notes_in(root / ".remember")
        stems: dict[str, list[Path]] = defaultdict(list)
        for path in vault.notes + vault.plans:
            stems[path.stem].append(path)
        vault.by_stem = dict(stems)
        return vault

    def rel(self, path: Path) -> str:
        """Return ``path`` relative to the vault root in POSIX form."""
        return path.relative_to(self.root).as_posix()

    def resolve(self, target: str) -> Path | None:
        """Return the file a wikilink target points at, or ``None``."""
        target = target.strip()
        if target.startswith("."):
            return None  # Obsidian never indexes dot folders, so it cannot link them
        if target.endswith(FILE_EXTENSIONS):
            candidate = self.root / target
            return candidate if candidate.exists() else None
        if "/" in target:
            candidate = self.root / f"{target}.md"
            return candidate if candidate.is_file() else None
        matches = self.by_stem.get(target)
        return matches[0] if matches else None

    def link_problem(self, match: re.Match[str]) -> str | None:
        """Return why a ``LINK`` match is broken, ``None`` when it resolves.

        A ``[[Note#Heading]]`` link must name a heading of the note (last ``#`` part).
        """
        target, heading = match.group(1).strip(), match.group(2)
        path = self.resolve(target)
        if path is None:
            return f"broken link [[{target}]]"
        if heading and path.suffix == ".md":
            wanted = heading.split("#")[-1].strip()
            text = strip_code(path.read_text(encoding="utf-8"))
            if wanted not in re.findall(r"(?m)^#{1,6} (.+?)\s*$", text):
                return f"broken heading link [[{target}#{wanted}]]"
        return None


def frontmatter(text: str) -> dict[str, Any]:
    """Return parsed YAML frontmatter, ``{}`` when absent (raises on bad YAML)."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}
    loaded = yaml.safe_load(match.group(1))
    return loaded if isinstance(loaded, dict) else {}


def has_frontmatter(text: str) -> bool:
    """Return whether ``text`` starts with a YAML frontmatter block."""
    return re.match(r"---\n.*?\n---\n", text, re.DOTALL) is not None


def strip_code(text: str) -> str:
    """Blank fenced and inline code (same length) so links inside are ignored."""
    chars = list(text)
    for pattern, flags in ((r"```.*?```", re.DOTALL), (r"`[^`\n]*`", 0)):
        for match in re.finditer(pattern, text, flags):
            for index in range(match.start(), match.end()):
                if chars[index] != "\n":
                    chars[index] = " "
    return "".join(chars)


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


def line_of(text: str, offset: int) -> int:
    """Return the 1-based line number of ``offset`` in ``text``."""
    return text.count("\n", 0, offset) + 1


def headings(text: str) -> list[str]:
    """Return ``## `` section titles."""
    return re.findall(r"(?m)^## (.+?)\s*$", text)


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


def comments_and_docstrings(path: Path) -> str:
    """Return a Python file's comments and triple-quoted strings, line-aligned."""
    text = path.read_text(encoding="utf-8")
    lines = [""] * (text.count("\n") + 2)
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            is_doc = token.type == tokenize.STRING and token.string.lstrip(
                "rRbBuUfF"
            ).startswith(('"""', "'''"))
            if token.type == tokenize.COMMENT or is_doc:
                start = token.start[0] - 1
                for offset, piece in enumerate(token.string.split("\n")):
                    lines[start + offset] += piece
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "\n".join(
            line if line.lstrip().startswith("#") else "" for line in text.splitlines()
        )
    return "\n".join(lines)


def module_docstring(path: Path) -> str:
    """Return the leading docstring source of a Python file (empty when none)."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r'\s*(?:#[^\n]*\n\s*)*[rRuU]?("""|\'\'\')(.*?)\1', text, re.DOTALL)
    return match.group(2) if match else ""


def is_placeholder(target: str) -> bool:
    """Return whether a link target is empty (Obsidian resolves everything else)."""
    return not target.strip()


def check_links(vault: Vault, path: Path, text: str) -> list[Finding]:
    """Return broken-wikilink findings for one Markdown file."""
    findings: list[Finding] = []
    for match in LINK.finditer(strip_code(text)):
        if is_placeholder(match.group(1)):
            continue
        problem = vault.link_problem(match)
        if problem:
            findings.append(
                Finding(vault.rel(path), line_of(text, match.start()), problem)
            )
    return findings


def check_markdown_links(vault: Vault, path: Path, text: str) -> list[Finding]:
    """Return findings for relative ``[text](target)`` links that point nowhere.

    Obsidian resolves them against the note's folder, then the vault root, adding
    ``.md`` when the target has no suffix; clicking an unresolved one creates a file.
    """
    findings: list[Finding] = []
    for match in MD_LINK.finditer(strip_code(text)):
        target = match.group(1)
        if "://" in target or target.startswith(("mailto:", "obsidian:")):
            continue
        candidates = [path.parent / target, vault.root / target]
        if not Path(target).suffix:
            candidates = [c.with_suffix(".md") for c in candidates] + candidates
        if not any(c.exists() for c in candidates):
            findings.append(
                Finding(
                    vault.rel(path),
                    line_of(text, match.start()),
                    f"broken markdown link ({target}); Obsidian would create a file",
                )
            )
    return findings


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


def check_note(
    vault: Vault,
    path: Path,
    glossary: dict[str, dict[str, Any]],
    emails: set[str],
) -> list[Finding]:
    """Return findings for one knowledge note."""
    rel = vault.rel(path)
    text = path.read_text(encoding="utf-8")
    findings = check_links(vault, path, text) + check_markdown_links(vault, path, text)
    findings += [Finding(rel, number, rule) for number, rule in scan_text(text, emails)]
    try:
        meta = frontmatter(text)
    except yaml.YAMLError as error:
        findings.append(
            Finding(rel, 1, f"frontmatter is not valid YAML ({type(error).__name__})")
        )
        return findings
    if not has_frontmatter(text):
        findings.append(Finding(rel, 1, "missing frontmatter"))
        return findings
    kind = meta.get("type")
    for key in REQUIRED_FIELDS:
        if key == "created" and kind == "generated":
            continue
        if key == "parent" and kind in ROOTLESS_TYPES:
            continue
        if not meta.get(key):
            findings.append(Finding(rel, 1, f"frontmatter has no '{key}'"))
    if kind not in NOTE_TYPES:
        findings.append(Finding(rel, 1, f"type '{kind}' not in {sorted(NOTE_TYPES)}"))
        return findings
    if kind in STATUSES and meta.get("status") not in STATUSES[kind]:
        findings.append(
            Finding(
                rel,
                1,
                f"status '{meta.get('status')}' not in {sorted(STATUSES[kind])}",
            )
        )
    for item in meta.get("related-code") or []:
        match = LINK.search(str(item))
        target = (match.group(1) if match else str(item)).strip()
        if not (vault.root / target).exists():
            findings.append(Finding(rel, 1, f"related-code target missing: {target}"))
    present = headings(text)
    findings += [
        Finding(rel, 1, f"missing section '## {heading}'")
        for heading in REQUIRED_SECTIONS.get(str(kind), [])
        if heading not in present and path.stem != FIXED_ISSUES
    ]
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
    findings += [
        Finding(rel, 1, f"abbreviation {item} used but never spelled out")
        for item in unexpanded(body, glossary)
    ]
    return findings


def check_isolation(vault: Vault) -> list[Finding]:
    """Return findings for knowledge notes no other hand-written note links to.

    Links from generated notes do not count: ``Decisions`` and ``Open issues``
    list every decision and issue, which would hide a note nobody refers to.
    """
    inbound: dict[Path, int] = defaultdict(int)
    for source in vault.notes + vault.plans + vault.memory:
        text = source.read_text(encoding="utf-8")
        if frontmatter_type(text) == "generated":
            continue
        text = strip_code(text)
        for match in LINK.finditer(text):
            target = vault.resolve(match.group(1))
            if target is not None and target != source:
                inbound[target] += 1
    return [
        Finding(vault.rel(path), 1, "isolated note: no other note links to it")
        for path in vault.notes
        if inbound[path] == 0
    ]


def frontmatter_type(text: str) -> object:
    """Return the ``type`` frontmatter value, ``None`` when absent or unparsable."""
    try:
        return frontmatter(text).get("type")
    except yaml.YAMLError:
        return None


def check_duplicates(vault: Vault) -> list[Finding]:
    """Return findings for note names shared by several notes (ambiguous links)."""
    return [
        Finding(
            vault.rel(paths[1]),
            1,
            f"note name '{stem}' also used by {vault.rel(paths[0])}",
        )
        for stem, paths in sorted(vault.by_stem.items())
        if len(paths) > 1 and stem.lower() != "readme"
    ]


def check_plans(vault: Vault) -> list[Finding]:
    """Return findings for ``plans/*.md`` frontmatter and headers."""
    findings: list[Finding] = []
    for path in vault.plans:
        if path.name.lower() == "readme.md":
            continue
        rel = vault.rel(path)
        text = path.read_text(encoding="utf-8")
        findings += check_links(vault, path, text)
        if not has_frontmatter(text):
            findings.append(
                Finding(rel, 1, "plan file has no frontmatter (type, status)", True)
            )
            continue
        try:
            meta = frontmatter(text)
        except yaml.YAMLError:
            findings.append(Finding(rel, 1, "frontmatter is not valid YAML"))
            continue
        if meta.get("type") not in NOTE_TYPES:
            findings.append(Finding(rel, 1, f"type '{meta.get('type')}' not allowed"))
        if meta.get("type") == "plan":
            if meta.get("status") not in STATUSES["plan"]:
                findings.append(
                    Finding(rel, 1, f"plan status '{meta.get('status')}' not valid")
                )
            findings += [
                Finding(rel, 1, f"plan header line '{line}' missing")
                for line in PLAN_HEADER
                if line not in text
            ]
    return findings


def code_files(root: Path) -> list[Path]:
    """Return package and test Python files checked for links and tags."""
    return sorted((root / "src" / "mllm_shap").rglob("*.py")) + sorted(
        (root / "tests").rglob("*.py")
    )


def check_code(vault: Vault, files: list[Path]) -> list[Finding]:
    """Return broken-link and traceability-tag findings for Python files."""
    findings: list[Finding] = []
    package = vault.root / "src" / "mllm_shap"
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
        is_test = path.name.startswith("test_")
        if not (in_package or is_test):
            continue
        if in_package and not TAG_DOMAIN.search(docstring):
            findings.append(Finding(rel, 1, "missing @domain tag", True))
        see = TAG_SEE.search(docstring)
        if see is None:
            findings.append(Finding(rel, 1, "missing @see [[...]] tag", True))
        elif in_package:
            target = vault.resolve(see.group(1))
            if target is None or target.suffix != ".md":
                findings.append(
                    Finding(rel, 1, f"@see target is not a note: {see.group(1)}", True)
                )
    return findings


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def architecture_note_for(root: Path, changed: str) -> Path | None:
    """Return the architecture note path for a changed ``src/mllm_shap`` file."""
    parts = Path(changed).parts
    if len(parts) < 3 or parts[:2] != ("src", "mllm_shap"):
        return None
    package = parts[2] if len(parts) > 3 else "core"
    return root / "knowledge" / "architecture" / f"{package}.md"


def check_freshness(vault: Vault) -> list[Finding]:
    """Return warnings for architecture notes older than their package's changes."""
    base = _git(vault.root, "merge-base", "HEAD", "origin/main")
    if not base:
        return []
    changed = _git(vault.root, "diff", "--name-only", f"{base}...HEAD") or ""
    notes: dict[Path, str] = {}
    for name in changed.splitlines():
        note = architecture_note_for(vault.root, name)
        if note is None:
            continue
        package_dir = str(Path(name).parent) if note.stem != "core" else name
        newest = _git(
            vault.root, "log", "-1", "--format=%cs", f"{base}..HEAD", "--", package_dir
        )
        if newest and newest > notes.get(note, ""):
            notes[note] = newest
    findings: list[Finding] = []
    for note, newest in sorted(notes.items()):
        rel = note.relative_to(vault.root).as_posix()
        if not note.is_file():
            findings.append(Finding(rel, 1, "architecture note missing", True))
            continue
        meta = frontmatter(note.read_text(encoding="utf-8"))
        stamp = str(meta.get("updated") or meta.get("created") or "")
        if isinstance(meta.get("updated"), date):
            stamp = meta["updated"].isoformat()
        if stamp < newest:
            findings.append(
                Finding(
                    rel,
                    1,
                    f"'updated' {stamp or 'missing'} is older than package change "
                    f"{newest}; update the note",
                    True,
                )
            )
    return findings


STRAY_SKIP = {".git", ".venv", ".obsidian", "_build", "node_modules", ".superpowers"}
STRAY_TOKENS = ("<%", "%>", "{{", "}}")


def check_stray_files(vault: Vault) -> list[Finding]:
    """Report files Obsidian creates when a link to a missing note is clicked.

    Such files carry an unexpanded Templater token in their name, are called
    ``Untitled``, or are empty Markdown outside ``knowledge/_templates``.
    """
    findings: list[Finding] = []
    for path in sorted(vault.root.rglob("*")):
        parts = set(path.relative_to(vault.root).parts)
        if parts & STRAY_SKIP or ("tmp" in parts and ".remember" in parts):
            continue
        name = path.name
        stray = any(token in name for token in STRAY_TOKENS) or name.startswith(
            "Untitled"
        )
        empty_note = (
            path.is_file()
            and path.suffix == ".md"
            and path.stat().st_size == 0
            and "_templates" not in parts
        )
        if stray or empty_note:
            findings.append(
                Finding(
                    vault.rel(path),
                    1,
                    "stray file created by Obsidian from a link to a missing note; "
                    "delete it and fix the link",
                )
            )
    return findings


def aux_markdown(vault: Vault) -> list[Path]:
    """Every Markdown file outside the notes and plans (checked on their own).

    Uses ``git ls-files`` (tracked and untracked, ``.gitignore`` honoured) and falls
    back to walking the tree without ``STRAY_SKIP`` folders outside a repository.
    """
    listed = _git(vault.root, "ls-files", "-co", "--exclude-standard")
    if listed is None:
        paths = [
            path.relative_to(vault.root).as_posix()
            for path in vault.root.rglob("*.md")
            # Templates are link-checked.
            if not set(path.relative_to(vault.root).parts) & STRAY_SKIP
        ]
    else:
        paths = listed.splitlines()
    own = {vault.rel(path) for path in [*vault.notes, *vault.plans]}
    return [
        vault.root / rel
        for rel in sorted(paths)
        if rel.endswith(".md") and rel not in own and (vault.root / rel).is_file()
    ]


def check_aux_links(vault: Vault) -> list[Finding]:
    """Return broken-wikilink findings for every other Markdown file.

    Obsidian creates an empty note when a broken link anywhere is clicked.
    """
    findings: list[Finding] = []
    for path in aux_markdown(vault):
        findings += check_links(vault, path, path.read_text(encoding="utf-8"))
    return findings


def run(
    root: Path, *, strict: bool = False, strict_tags: bool = False
) -> tuple[list[Finding], int, int]:
    """Run every check; return (findings, note count, code file count)."""
    vault = Vault.load(root)
    glossary = load_glossary(root)
    emails = maintainer_emails(root)
    findings: list[Finding] = check_stray_files(vault) + check_aux_links(vault)
    for path in vault.notes:
        findings += check_note(vault, path, glossary, emails)
    for path in vault.memory:
        findings += [
            Finding(vault.rel(path), number, rule)
            for number, rule in scan_text(path.read_text(encoding="utf-8"), emails)
        ]
    findings += check_duplicates(vault)
    findings += check_isolation(vault)
    plan_findings = check_plans(vault)
    files = code_files(root)
    code_findings = check_code(vault, files)
    fresh = check_freshness(vault)
    if strict:
        plan_findings = [_promote(item) for item in plan_findings]
        fresh = [_promote(item) for item in fresh]
    if strict_tags:
        code_findings = [_promote(item) for item in code_findings]
    findings += plan_findings + code_findings + fresh
    return findings, len(vault.notes), len(files)


def _promote(finding: Finding) -> Finding:
    return Finding(finding.path, finding.line, finding.message, warning=False)


def iter_errors(findings: list[Finding]) -> Iterator[Finding]:
    """Yield findings that are errors."""
    return (item for item in findings if not item.warning)


def main(argv: list[str] | None = None) -> int:
    """Run the vault checks and print findings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print errors only")
    parser.add_argument("--strict", action="store_true", help="CI mode")
    parser.add_argument("--strict-tags", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    findings, notes, files = run(
        args.root.resolve(), strict=args.strict, strict_tags=args.strict_tags
    )
    errors = list(iter_errors(findings))
    warnings = [item for item in findings if item.warning]
    for item in errors:
        print(item)
    if not args.quiet:
        if len(warnings) > 20:
            for item in warnings[:20]:
                print(item)
            print(f"... and {len(warnings) - 20} more warning(s)")
        else:
            for item in warnings:
                print(item)
    if errors:
        print(f"check_vault: {len(errors)} issue(s)")
        return 1
    if not args.quiet:
        print(
            f"check_vault: ok ({notes} notes, {files} code files, "
            f"{len(warnings)} warning(s))"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
