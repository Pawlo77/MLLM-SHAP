"""Consistency checks for the Obsidian vault in the repository root.

Usage:
    python scripts/check_vault.py [--quiet] [--strict] [--strict-tags] [--root DIR]

Prints ``path:line: message`` per finding and exits 1 on any error. Shared checks:
wikilinks and Markdown links, stray files Obsidian creates, isolated and duplicate
notes, secrets (``secrets = false`` skips them) and the frontmatter schema, all set in
``[tool.agent-kit.vault]``. Project rules live in ``scripts/vault_rules.py``
(``check_file``, ``check_vault``, ``code_files``). ``--strict`` promotes warnings of
group ``strict``, ``--strict-tags`` those of ``tags``.
"""

import argparse
import importlib.util
import io
import re
import subprocess  # nosec B404
import sys
import tokenize
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "hooks"))

from check_memory_files import maintainer_emails, scan_text  # noqa: E402
from kit_config import load as load_config  # noqa: E402

DEFAULTS: dict[str, Any] = {
    "notes": ["knowledge"],
    "plans": ["plans"],
    "memory": [".remember"],
    "flat": ["plans", ".remember"],
    "skip-dirs": ["_templates"],
    "skip-names": ["AGENTS.md", "CLAUDE.md"],
    "isolated": ["knowledge"],
    "link-dirs": [],
    "stray-skip": [],
    "secrets": True,
}
"""Vault layout used when ``[tool.agent-kit.vault]`` leaves a key out."""

AGENT_FILES = frozenset(DEFAULTS["skip-names"])
"""Agent instructions beside the notes (``CLAUDE.md`` links to ``AGENTS.md``)."""

SKIPPED_FOLDERS = frozenset(DEFAULTS["skip-dirs"])
"""Folders under the note folders that hold no notes: Templater templates."""

LINK = re.compile(r"\[\[([^\]|#\\]+)(?:#([^\]|\\]*))?(?:\\?\|[^\]]*)?\]\]")
"""Wikilink target and optional heading (``\\|`` is the escaped alias pipe)."""

MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
"""Markdown link target; Obsidian resolves relative ones like wikilinks."""

FILE_EXTENSIONS = (
    *(".py", ".md", ".pdf", ".ipynb", ".yaml", ".yml", ".toml", ".json"),
    *(".rst", ".txt", ".cfg", ".sh"),
)
"""Link targets with these suffixes are vault paths to files."""

STRAY_SKIP = {
    *(".git", ".venv", ".obsidian", "_build", "node_modules", ".superpowers"),
    *(".mypy_cache", ".ruff_cache", ".pytest_cache", "__pycache__"),
}
"""Folders the stray-file walk never enters."""

STRAY_TOKENS = ("<%", "%>", "{{", "}}")
"""Unexpanded template tokens in a file name mark a stray Obsidian file."""


@dataclass(frozen=True)
class Finding:
    """One checker result; ``group`` says which strict flag promotes a warning."""

    path: str
    line: int
    message: str
    warning: bool = False
    group: str = ""

    def __str__(self) -> str:
        """Format the issue as ``path:line: message``."""
        level = "warning: " if self.warning else ""
        return f"{self.path}:{self.line}: {level}{self.message}"


def settings(root: Path) -> dict[str, Any]:
    """Return ``[tool.agent-kit.vault]`` over :data:`DEFAULTS`."""
    table = load_config(root).get("vault", {})
    return {**DEFAULTS, **(table if isinstance(table, dict) else {})}


def _collect(root: Path, folders: list[str], cfg: dict[str, Any]) -> list[Path]:
    """Return the notes of ``folders`` in order; direct children for ``flat`` ones."""
    found: list[Path] = []
    for name in folders:
        folder = root / name
        pattern = "*.md" if name in cfg["flat"] else "**/*.md"
        found += sorted(
            path
            for path in folder.glob(pattern)
            if path.name not in cfg["skip-names"]
            and not set(cfg["skip-dirs"]) & set(path.relative_to(root).parts)
        )
    return found


@dataclass
class Vault:
    """Notes and link targets of a vault rooted at ``root``."""

    root: Path
    cfg: dict[str, Any] = field(default_factory=lambda: dict(DEFAULTS))
    notes: list[Path] = field(default_factory=list)
    plans: list[Path] = field(default_factory=list)
    memory: list[Path] = field(default_factory=list)
    by_stem: dict[str, list[Path]] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "Vault":
        """Collect notes, plans and memory notes under ``root``."""
        cfg = settings(root)
        vault = cls(root=root, cfg=cfg)
        vault.notes = _collect(root, cfg["notes"], cfg)
        vault.plans = _collect(root, cfg["plans"], cfg)
        vault.memory = _collect(root, cfg["memory"], cfg)
        stems: dict[str, list[Path]] = defaultdict(list)
        targets = _collect(root, cfg["link-dirs"], cfg)  # linkable, not checked
        for path in vault.notes + vault.plans + targets:
            stems[path.stem].append(path)
        vault.by_stem = dict(stems)
        return vault

    def rel(self, path: Path) -> str:
        """Return ``path`` relative to the vault root in POSIX form."""
        return path.relative_to(self.root).as_posix()

    def under(self, path: Path, *folders: str) -> bool:
        """Return whether ``path`` lies in one of the vault ``folders``."""
        return any(path.is_relative_to(self.root / name) for name in folders)

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


def frontmatter_type(text: str) -> object:
    """Return the ``type`` frontmatter value, ``None`` when absent or unparsable."""
    try:
        return frontmatter(text).get("type")
    except yaml.YAMLError:
        return None


def strip_code(text: str) -> str:
    """Blank fenced and inline code (same length) so links inside are ignored."""
    chars = list(text)
    for pattern, flags in ((r"```.*?```", re.DOTALL), (r"`[^`\n]*`", 0)):
        for match in re.finditer(pattern, text, flags):
            for index in range(match.start(), match.end()):
                if chars[index] != "\n":
                    chars[index] = " "
    return "".join(chars)


def line_of(text: str, offset: int) -> int:
    """Return the 1-based line number of ``offset`` in ``text``."""
    return text.count("\n", 0, offset) + 1


def headings(text: str) -> list[str]:
    """Return ``## `` section titles."""
    return re.findall(r"(?m)^## (.+?)\s*$", text)


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
    """Return whether a link target is a template placeholder, not a real link."""
    target = target.strip()
    return not target or "{{" in target or target in {"...", "…"}


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
        target = match.group(1).replace("%20", " ")
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


def check_schema(vault: Vault, path: Path, text: str) -> list[Finding]:
    """Return frontmatter findings for a note in the schema folders.

    ``[tool.agent-kit.vault.schema]`` sets ``dirs``, ``required`` keys, allowed
    ``types``, per-type ``statuses`` and ``sections``, ``exempt`` types per key,
    ``sections-exempt`` note names and ``link-fields`` whose targets must exist.
    """
    schema = vault.cfg.get("schema", {})
    if not schema or not vault.under(path, *schema.get("dirs", [])):
        return []
    rel = vault.rel(path)
    try:
        meta = frontmatter(text)
    except yaml.YAMLError as error:
        return [
            Finding(rel, 1, f"frontmatter is not valid YAML ({type(error).__name__})")
        ]
    if not has_frontmatter(text):
        return [Finding(rel, 1, "missing frontmatter")]
    kind = meta.get("type")
    exempt = schema.get("exempt", {})
    findings = [
        Finding(rel, 1, f"frontmatter has no '{key}'")
        for key in schema.get("required", [])
        if not meta.get(key) and kind not in exempt.get(key, [])
    ]
    types = schema.get("types", [])
    if types and kind not in types:
        return [*findings, Finding(rel, 1, f"type '{kind}' not in {sorted(types)}")]
    statuses = schema.get("statuses", {}).get(str(kind))
    if statuses and meta.get("status") not in statuses:
        findings.append(
            Finding(rel, 1, f"status '{meta.get('status')}' not in {sorted(statuses)}")
        )
    for key in schema.get("link-fields", []):
        for item in meta.get(key) or []:
            match = LINK.search(str(item))
            target = (match.group(1) if match else str(item)).strip()
            if not (vault.root / target).exists():
                findings.append(Finding(rel, 1, f"{key} target missing: {target}"))
    if path.stem not in schema.get("sections-exempt", []):
        present = headings(text)
        findings += [
            Finding(rel, 1, f"missing section '## {heading}'")
            for heading in schema.get("sections", {}).get(str(kind), [])
            if heading not in present
        ]
    return findings


def check_secrets(
    vault: Vault, path: Path, text: str, emails: set[str]
) -> list[Finding]:
    """Return secret and personal-data findings for one file."""
    return [Finding(vault.rel(path), n, rule) for n, rule in scan_text(text, emails)]


def check_isolation(vault: Vault) -> list[Finding]:
    """Return findings for notes in the ``isolated`` folders nobody links to.

    Links from generated notes do not count: they list every decision and issue,
    which would hide a note nobody refers to. Generated notes need no inbound link.
    """
    inbound: dict[Path, int] = defaultdict(int)
    for source in vault.notes + vault.plans + vault.memory:
        text = source.read_text(encoding="utf-8")
        if frontmatter_type(text) == "generated":
            continue
        for match in LINK.finditer(strip_code(text)):
            target = vault.resolve(match.group(1))
            if target is not None and target != source:
                inbound[target] += 1
    return [
        Finding(vault.rel(path), 1, "isolated note: no other note links to it")
        for path in vault.notes
        if vault.under(path, *vault.cfg["isolated"])
        and inbound[path] == 0
        and frontmatter_type(path.read_text(encoding="utf-8")) != "generated"
    ]


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


def git(root: Path, *args: str) -> str | None:
    """Return the output of ``git *args`` in ``root``, ``None`` on failure."""
    try:
        result = subprocess.run(  # nosec B603 B607
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _walk(root: Path, skip: set[str]) -> Iterator[Path]:
    """Yield files and folders under ``root``, pruning the ``skip`` folders."""
    for folder, dirs, names in root.walk():
        dirs[:] = [name for name in dirs if name not in skip]
        yield from (folder / name for name in [*dirs, *names])


def check_stray_files(vault: Vault) -> list[Finding]:
    """Report files Obsidian creates when a link to a missing note is clicked.

    Such files carry an unexpanded Templater token in their name, are called
    ``Untitled``, or are empty Markdown outside the ``skip-dirs`` template folders.
    """
    skip = STRAY_SKIP | set(vault.cfg["stray-skip"])
    findings: list[Finding] = []
    for path in sorted(_walk(vault.root, skip)):
        parts = set(path.relative_to(vault.root).parts)
        if "tmp" in parts and ".remember" in parts:
            continue
        stray = any(token in path.name for token in STRAY_TOKENS)
        stray = stray or path.name.startswith("Untitled")
        empty_note = (
            path.is_file()
            and path.suffix == ".md"
            and path.stat().st_size == 0
            and not set(vault.cfg["skip-dirs"]) & parts
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
    back to walking the tree without the stray-skip folders outside a repository.
    """
    listed = git(vault.root, "ls-files", "-co", "--exclude-standard")
    if listed is None:
        skip = STRAY_SKIP | set(vault.cfg["stray-skip"])
        paths = [
            path.relative_to(vault.root).as_posix()
            for path in vault.root.rglob("*.md")
            if not set(path.relative_to(vault.root).parts) & skip
        ]
    else:
        paths = listed.splitlines()
    own = {vault.rel(path) for path in [*vault.notes, *vault.plans]}
    return [
        vault.root / rel
        for rel in sorted(paths)
        if rel.endswith(".md") and rel not in own and (vault.root / rel).is_file()
    ]


def load_rules() -> Any:
    """Return the project's ``scripts/vault_rules.py`` module, ``None`` without one."""
    path = SCRIPTS / "vault_rules.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("vault_rules", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vault_rules"] = module
    spec.loader.exec_module(module)
    return module


def _promote(finding: Finding) -> Finding:
    return Finding(finding.path, finding.line, finding.message, warning=False)


def run(
    root: Path, *, strict: bool = False, strict_tags: bool = False
) -> tuple[list[Finding], int, int]:
    """Run every check; return (findings, note count, code file count)."""
    vault = Vault.load(root)
    rules = load_rules()
    emails = maintainer_emails(root)
    findings = check_stray_files(vault)
    for path in aux_markdown(vault):
        findings += check_links(vault, path, path.read_text(encoding="utf-8"))
    for path in vault.notes + vault.plans:
        text = path.read_text(encoding="utf-8")
        findings += check_links(vault, path, text)
        findings += check_markdown_links(vault, path, text)
        findings += check_schema(vault, path, text)
        if vault.cfg["secrets"] and path in vault.notes:
            findings += check_secrets(vault, path, text, emails)
        if rules is not None:
            findings += rules.check_file(vault, path, text)
    for path in vault.memory if vault.cfg["secrets"] else []:
        text = path.read_text(encoding="utf-8")
        findings += check_secrets(vault, path, text, emails)
    findings += check_duplicates(vault) + check_isolation(vault)
    count = 0
    if rules is not None:
        findings += rules.check_vault(vault)
        count = len(rules.code_files(root)) if hasattr(rules, "code_files") else 0
    promoted = {"strict": strict, "tags": strict_tags}
    findings = [
        _promote(item) if item.warning and promoted.get(item.group) else item
        for item in findings
    ]
    return findings, len(vault.notes), count


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
        for item in warnings[:20]:
            print(item)
        if len(warnings) > 20:
            print(f"... and {len(warnings) - 20} more warning(s)")
    if errors:
        print(f"check_vault: {len(errors)} issue(s)")
        return 1
    if not args.quiet:
        code = f", {files} code files" if files else ""
        print(f"check_vault: ok ({notes} notes{code}, {len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
