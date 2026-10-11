"""Skills and subagents have the frontmatter their loaders require.

The skill checks re-implement the Agent Skills specification validator
(``skills-ref`` 0.1.1, ``skills_ref/validator.py``; https://agentskills.io/specification)
with one difference: the client extension keys in ``CLIENT_FIELDS`` are allowed, because
Claude Code and Cursor read them and the upstream validator rejects them.

@see [[knowledge/handbook/Onboarding]]
"""

import json
import re
import unicodedata
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
AGENTS = ROOT / ".claude" / "agents"

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

SPEC_FIELDS = {
    "name",
    "description",
    "license",
    "allowed-tools",
    "metadata",
    "compatibility",
}
"""Frontmatter keys the Agent Skills specification allows."""

CLIENT_FIELDS = {"paths", "disable-model-invocation"}
"""Client extensions in use: ``paths`` (Cursor/Claude Code file scoping) and
``disable-model-invocation`` (Claude Code: run only when invoked by name)."""

MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_COMPATIBILITY = 500


def _frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?\n)---\n(.*)$", text, re.DOTALL)
    assert match is not None, f"{path}: no frontmatter"
    meta = yaml.safe_load(match.group(1))
    assert isinstance(meta, dict), f"{path}: frontmatter is not a mapping"
    return meta, match.group(2)


def spec_errors(meta: dict[str, object], directory: str) -> list[str]:
    """Return Agent Skills specification violations for one skill's frontmatter."""
    errors: list[str] = []
    extra = set(meta) - SPEC_FIELDS - CLIENT_FIELDS
    if extra:
        errors.append(f"unexpected fields: {sorted(extra)}")
    name = meta.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("name must be a non-empty string")
    else:
        name = unicodedata.normalize("NFKC", name.strip())
        if len(name) > MAX_NAME:
            errors.append(f"name longer than {MAX_NAME}")
        if not NAME_RE.match(name):
            errors.append("name must be lowercase letters, digits and single hyphens")
        if unicodedata.normalize("NFKC", directory) != name:
            errors.append(f"name {name!r} differs from directory {directory!r}")
    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("description must be a non-empty string")
    elif len(description) > MAX_DESCRIPTION:
        errors.append(f"description longer than {MAX_DESCRIPTION}")
    compatibility = meta.get("compatibility", "")
    if not isinstance(compatibility, str) or len(compatibility) > MAX_COMPATIBILITY:
        errors.append(f"compatibility must be a string of <= {MAX_COMPATIBILITY}")
    if "metadata" in meta:
        metadata = meta["metadata"]
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        ):
            errors.append("metadata must map strings to strings")
    if "allowed-tools" in meta and not isinstance(meta["allowed-tools"], str):
        errors.append("allowed-tools must be a space-delimited string")
    return errors


SKILL_FILES = sorted(p / "SKILL.md" for p in SKILLS.iterdir() if p.is_dir())
AGENT_FILES = sorted(p for p in AGENTS.glob("*.md") if p.name != "VENDORED.md")


@pytest.mark.parametrize("path", SKILL_FILES, ids=lambda p: p.parent.name)
def test_skill_frontmatter(path: Path) -> None:
    """Agent Skills spec: valid name and description, known keys, short body."""
    meta, body = _frontmatter(path)
    assert spec_errors(meta, path.parent.name) == []
    assert len(body.splitlines()) < 500


@pytest.mark.parametrize(
    ("meta", "fragment"),
    [
        ({"name": "Bad-Name", "description": "x"}, "lowercase"),
        ({"name": "a--b", "description": "x"}, "single hyphens"),
        ({"name": "-a", "description": "x"}, "single hyphens"),
        ({"name": "other", "description": "x"}, "differs from directory"),
        ({"name": "demo", "description": ""}, "description"),
        ({"name": "demo", "description": "x" * 1025}, "longer than 1024"),
        ({"name": "demo", "description": "x", "argument-hint": "y"}, "unexpected"),
        ({"name": "demo", "description": "x", "metadata": {"a": 1}}, "metadata"),
        ({"name": "demo", "description": "x", "compatibility": "c" * 501}, "compat"),
    ],
)
def test_spec_errors_catch_violations(meta: dict[str, object], fragment: str) -> None:
    """Each specification rule rejects a violating frontmatter."""
    errors = spec_errors(meta, "demo")
    assert any(fragment in error for error in errors), errors


def test_spec_errors_accept_client_fields() -> None:
    """The documented client extensions pass; a clean skill has no errors."""
    meta = {
        "name": "demo",
        "description": "Does a thing. Use when asked.",
        "paths": ["src/**"],
        "disable-model-invocation": True,
        "license": "Apache-2.0",
    }
    assert spec_errors(meta, "demo") == []


EVAL_FILES = sorted(SKILLS.glob("*/evals/trigger-evals.json"))


def test_trigger_evals_exist_for_review_and_bugspot() -> None:
    """The review and bug-hunting skills carry trigger evals (skill-creator format)."""
    names = {path.parents[1].name for path in EVAL_FILES}
    assert {"review", "bugspot"} <= names


@pytest.mark.parametrize("path", EVAL_FILES, ids=lambda p: p.parents[1].name)
def test_trigger_evals_shape(path: Path) -> None:
    """Each eval is a query with a boolean should_trigger; both outcomes appear."""
    items = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(items, list)
    assert items
    for item in items:
        assert set(item) == {"query", "should_trigger"}
        assert isinstance(item["query"], str)
        assert item["query"].strip()
        assert isinstance(item["should_trigger"], bool)
    assert {item["should_trigger"] for item in items} == {True, False}


def test_canonical_skill_links() -> None:
    """Claude Code and Cursor see the canonical skills through symlinks."""
    for link in (ROOT / ".claude" / "skills", ROOT / ".cursor" / "skills"):
        assert link.is_symlink()
        assert link.resolve() == SKILLS.resolve()
    commands = ROOT / ".claude" / "commands"
    assert commands.is_symlink()
    assert commands.resolve() == (ROOT / ".cursor" / "commands").resolve()


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.stem)
def test_agent_frontmatter(path: Path) -> None:
    """Subagents name themselves, declare tools and model, and stay read-only."""
    meta, body = _frontmatter(path)
    assert meta.get("name") == path.stem
    for key in ("description", "tools", "model"):
        assert isinstance(meta.get(key), str), key
        assert str(meta[key]).strip(), key
    tools = {t.strip() for t in str(meta["tools"]).split(",")}
    assert not tools & WRITE_TOOLS, f"{path.stem} must be read-only"
    assert body.strip()
