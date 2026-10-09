"""Guard committed memory and vault notes against secrets and personal data.

Usage:
    python scripts/hooks/check_memory_files.py [FILE ...]

Checks the given files, else every ``.remember/*.md`` and ``knowledge/**/*.md``; prints
``path:line: <rule>`` per finding and exits 1 on any. Allowed e-mail addresses: the
``pyproject.toml`` authors, ``noreply`` and ``example.com``/``.org`` (never the local
``git config user.email``); loopback IPv4 addresses are allowed.

@domain tooling
"""

import re
import sys
import tomllib
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential assignment",
        re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S{8,}"),
    ),
    ("OpenAI-style secret key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("GitHub personal access token", re.compile(r"ghp_[A-Za-z0-9]{30,}")),
    ("PEM block", re.compile(r"-----BEGIN")),
    ("bearer token", re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}")),
)
"""(rule name, pattern) pairs that flag a line on any match."""

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
"""E-mail address pattern."""

IPV4 = re.compile(r"(?<![\w.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\w.])")
"""Dotted-quad IPv4 pattern (not part of a longer dotted version string)."""


def maintainer_emails(root: Path = ROOT) -> set[str]:
    """Return the maintainer's public addresses from ``pyproject.toml``."""
    emails: set[str] = set()
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        project = tomllib.loads(pyproject.read_text(encoding="utf-8")).get(
            "project", {}
        )
        for key in ("authors", "maintainers"):
            for person in project.get(key, []):
                if isinstance(person, dict) and person.get("email"):
                    emails.add(str(person["email"]).lower())
    return emails


def _email_allowed(address: str, allowed: set[str]) -> bool:
    lowered = address.lower()
    local, _, domain = lowered.partition("@")
    return (
        lowered in allowed
        or local == "noreply"
        or domain.endswith("users.noreply.github.com")
        or domain in {"example.com", "example.org"}
    )


def _ip_allowed(octets: tuple[str, ...]) -> bool:
    values = [int(value) for value in octets]
    if any(value > 255 for value in values):
        return True  # not an IPv4 address (e.g. a version-like number)
    return values[0] == 127


def check_line(line: str, allowed_emails: set[str]) -> list[str]:
    """Return the rule names a single line violates."""
    found = [name for name, pattern in SECRET_RULES if pattern.search(line)]
    found.extend(
        f"e-mail address {match.group(0)}"
        for match in EMAIL.finditer(line)
        if not _email_allowed(match.group(0), allowed_emails)
    )
    found.extend(
        f"non-loopback IPv4 address {match.group(0)}"
        for match in IPV4.finditer(line)
        if not _ip_allowed(match.groups())
    )
    return found


def check_file(path: Path, allowed_emails: set[str], root: Path = ROOT) -> list[str]:
    """Return ``path:line: rule`` findings for one file."""
    try:
        rel = path.resolve().relative_to(root)
    except ValueError:
        rel = path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    return [
        f"{rel}:{number}: {rule}" for number, rule in scan_text(text, allowed_emails)
    ]


def scan_text(text: str, allowed_emails: set[str]) -> list[tuple[int, str]]:
    """Return ``(line number, rule)`` findings for ``text``."""
    return [
        (number, rule)
        for number, line in enumerate(text.splitlines(), start=1)
        for rule in check_line(line, allowed_emails)
    ]


def default_files(root: Path = ROOT) -> Iterator[Path]:
    """Yield every memory and vault Markdown file under ``root``."""
    yield from sorted((root / ".remember").glob("*.md"))
    yield from sorted((root / "knowledge").rglob("*.md"))


def in_scope(path: Path, root: Path = ROOT) -> bool:
    """Return whether ``path`` is a memory or vault Markdown file."""
    try:
        rel = path.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    return rel.endswith(".md") and rel.startswith((".remember/", "knowledge/"))


def run(paths: Iterable[Path], root: Path = ROOT) -> list[str]:
    """Check ``paths`` and return all findings."""
    allowed = maintainer_emails(root)
    findings: list[str] = []
    for path in paths:
        findings.extend(check_file(path, allowed, root))
    return findings


def main(argv: list[str]) -> int:
    """Run the guard over ``argv`` files or the default set."""
    paths = (
        [Path(arg) for arg in argv if in_scope(Path(arg))]
        if argv
        else list(default_files())
    )
    findings = run(paths)
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
