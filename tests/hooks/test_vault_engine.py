"""The shared vault checker: layout from pyproject, shared checks and project rules.

@see [[scripts/check_vault.py]]
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

NOTE = '---\ntitle: "{0}"\ntype: {1}\nparent: "[[Map]]"\n---\n\n# {0}\n\n{2}\n'


@pytest.fixture(scope="module")
def checker(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    """Load scripts/check_vault.py without the project's own rules."""
    spec = importlib.util.spec_from_file_location(
        "check_vault_under_test", ROOT / "scripts" / "check_vault.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPTS = tmp_path_factory.mktemp("no-rules")  # the shared checks only
    return module


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """A valid vault: a map linking one note, under the default layout."""
    _write(tmp_path, "pyproject.toml", "[project]\nname = 'demo'\n")
    _write(tmp_path, "knowledge/Map.md", NOTE.format("Map", "map", "[[Note]]"))
    _write(tmp_path, "knowledge/Note.md", NOTE.format("Note", "concept", "Text."))
    return tmp_path


def _errors(checker: ModuleType, root: Path, **flags: bool) -> list[str]:
    findings, _, _ = checker.run(root, **flags)
    return [str(item) for item in findings if not item.warning]


def test_valid_vault_passes(checker: ModuleType, vault: Path) -> None:
    assert _errors(checker, vault) == []


def test_project_without_notes_passes(checker: ModuleType, tmp_path: Path) -> None:
    assert _errors(checker, tmp_path) == []


@pytest.mark.parametrize(
    ("rel", "text", "expected"),
    [
        ("knowledge/Note.md", "[[Missing]]", "broken link [[Missing]]"),
        ("knowledge/Note.md", "[[Map#Nope]]", "broken heading link"),
        ("knowledge/Note.md", "[x](gone/file.md)", "broken markdown link"),
        ("knowledge/Lonely.md", "Text.", "isolated note"),
        ("knowledge/sub/Note.md", "Text.", "also used by"),
        ("knowledge/Note.md", "api_key = abcdefghijklmnop123", "credential"),
        ("Untitled.md", "", "stray file"),
        ("knowledge/Note <% x %>.md", "Text.", "stray file"),
    ],
)
def test_shared_rules_bite(
    checker: ModuleType, vault: Path, rel: str, text: str, expected: str
) -> None:
    path = vault / rel
    if path.exists() and path.stat().st_size:
        text = path.read_text(encoding="utf-8") + "\n" + text
    _write(vault, rel, text)
    assert any(expected in line for line in _errors(checker, vault)), rel


def test_layout_comes_from_pyproject(checker: ModuleType, vault: Path) -> None:
    _write(vault, "docs/Doc.md", NOTE.format("Doc", "concept", "Text."))
    assert _errors(checker, vault) == []  # docs/ is not a note folder by default
    table = (
        '[tool.agent-kit.vault]\nnotes = ["knowledge", "docs"]\nisolated = ["docs"]\n'
    )
    _write(vault, "pyproject.toml", "[project]\nname = 'demo'\n" + table)
    assert any("isolated note" in line for line in _errors(checker, vault))


def test_link_dirs_are_link_targets_only(checker: ModuleType, vault: Path) -> None:
    _write(vault, "venues/Venue.md", "A venue note with no frontmatter.\n")
    _write(vault, "knowledge/Note.md", NOTE.format("Note", "concept", "[[Venue]]"))
    assert any("broken link [[Venue]]" in line for line in _errors(checker, vault))
    table = '[tool.agent-kit.vault]\nlink-dirs = ["venues"]\n'
    _write(vault, "pyproject.toml", "[project]\nname = 'demo'\n" + table)
    assert _errors(checker, vault) == []


def test_secrets_scan_can_be_turned_off(checker: ModuleType, vault: Path) -> None:
    _write(
        vault,
        "knowledge/Note.md",
        NOTE.format("Note", "concept", "token = abcdefghijkl1234"),
    )
    assert any("credential" in line for line in _errors(checker, vault))
    table = "[tool.agent-kit.vault]\nsecrets = false\n"
    _write(vault, "pyproject.toml", "[project]\nname = 'demo'\n" + table)
    assert _errors(checker, vault) == []


SCHEMA = """
[tool.agent-kit.vault.schema]
dirs = ["knowledge"]
required = ["title", "type", "parent"]
types = ["map", "concept", "issue"]
sections-exempt = ["Exempt"]

[tool.agent-kit.vault.schema.exempt]
parent = ["map"]

[tool.agent-kit.vault.schema.statuses]
issue = ["open", "fixed"]

[tool.agent-kit.vault.schema.sections]
issue = ["Symptom", "Fix"]
"""


@pytest.mark.parametrize(
    ("note", "expected"),
    [
        ("---\ntitle: N\n---\n\n# N\n", "frontmatter has no 'type'"),
        ("# N\n", "missing frontmatter"),
        ("---\ntitle: [x\n---\n", "not valid YAML"),
        ("---\ntitle: N\ntype: bogus\nparent: x\n---\n", "type 'bogus' not in"),
        (
            "---\ntitle: N\ntype: issue\nparent: x\nstatus: closed\n---\n",
            "status 'closed'",
        ),
        ("---\ntitle: N\ntype: issue\nparent: x\nstatus: open\n---\n", "## Symptom"),
    ],
)
def test_schema_rules_bite(
    checker: ModuleType, vault: Path, note: str, expected: str
) -> None:
    _write(vault, "pyproject.toml", "[project]\nname = 'demo'\n" + SCHEMA)
    _write(vault, "knowledge/Note.md", note + "\n[[Map]]\n")
    assert any(expected in line for line in _errors(checker, vault)), note


def test_schema_exemptions(checker: ModuleType, vault: Path) -> None:
    _write(vault, "pyproject.toml", "[project]\nname = 'demo'\n" + SCHEMA)
    _write(
        vault,
        "knowledge/Map.md",
        NOTE.format("Map", "map", "[[Note]]").replace('parent: "[[Map]]"\n', ""),
    )
    _write(vault, "knowledge/Note.md", NOTE.format("Note", "concept", "x"))
    assert _errors(checker, vault) == []


def test_project_rules_and_strict_groups(
    checker: ModuleType, vault: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / "scripts"
    _write(
        scripts,
        "vault_rules.py",
        "from check_vault import Finding\n\n"
        "def check_file(vault, path, text):\n"
        "    return [Finding(vault.rel(path), 1, 'rule', True, 'strict')]"
        " if path.stem == 'Note' else []\n\n"
        "def check_vault(vault):\n"
        "    return [Finding('x', 1, 'tag', True, 'tags')]\n",
    )
    monkeypatch.setattr(checker, "SCRIPTS", scripts)
    assert _errors(checker, vault) == []
    assert _errors(checker, vault, strict=True) == ["knowledge/Note.md:1: rule"]
    assert _errors(checker, vault, strict_tags=True) == ["x:1: tag"]
