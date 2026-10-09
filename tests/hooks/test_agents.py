"""Subagent roster: frontmatter, report contracts, attribution and the Codex copies.

@see [[scripts/sync_agents.py]]
"""

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / ".claude" / "agents"
AGENT_FILES = sorted(p for p in AGENTS.glob("*.md") if p.name != "VENDORED.md")
VENDORED = {"docs-auditor", "pr-writer"}
"""Agents vendored (in whole or, for docs-auditor, in part) under VENDORED.md."""


@pytest.fixture(scope="module")
def sync_agents() -> ModuleType:
    """Load scripts/sync_agents.py without installing it as a package."""
    path = ROOT / "scripts" / "sync_agents.py"
    spec = importlib.util.spec_from_file_location("sync_agents", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(path: Path) -> tuple[dict[str, object], str]:
    _, head, body = path.read_text(encoding="utf-8").split("---\n", 2)
    meta = yaml.safe_load(head)
    assert isinstance(meta, dict)
    return meta, body


@pytest.mark.parametrize("path", AGENT_FILES, ids=lambda p: p.stem)
def test_agent_has_fields_and_report_contract(path: Path) -> None:
    """Every agent declares name, description, tools, model and a report format."""
    meta, body = _split(path)
    for key in ("name", "description", "tools", "model"):
        assert isinstance(meta.get(key), str), key
        assert str(meta[key]).strip(), key
    assert "\n## Report format\n" in "\n" + body


def test_vendored_agents_are_attributed() -> None:
    """Each vendored agent is listed in VENDORED.md with a pinned commit."""
    text = (AGENTS / "VENDORED.md").read_text(encoding="utf-8")
    for name in sorted(VENDORED):
        assert (AGENTS / f"{name}.md").is_file(), name
        assert name in text, name
    assert "3e2fbb6bb6fac2eb65d3350ef1f5398ee3213f0f" in text
    assert "aecd4c852f10b466245f18383fa6aad8c0b10d57" in text
    assert "trailofbits" not in {p.stem for p in AGENT_FILES}


def test_codex_agents_are_current(sync_agents: ModuleType) -> None:
    """The committed .codex/agents/*.toml match the Markdown sources."""
    assert sync_agents.main(["--check"]) == 0


def test_codex_agent_round_trip(sync_agents: ModuleType) -> None:
    """Generated TOML parses and carries the body verbatim; sandbox follows tools."""
    for path, text in sync_agents.build(ROOT).items():
        data = tomllib.loads(text)
        source = AGENTS / f"{path.stem}.md"
        meta, body = sync_agents.split_agent(source.read_text(encoding="utf-8"))
        assert data["name"] == path.stem
        assert data["description"] == str(meta["description"]).strip()
        assert data["developer_instructions"] == body
        assert not sync_agents.tools_of(meta) & sync_agents.WRITE_TOOLS, path.stem
        expected = meta.get("codex-sandbox", "read-only")
        assert data["sandbox_mode"] == expected, path.stem


def test_make_running_agents_opt_out_of_read_only(sync_agents: ModuleType) -> None:
    """Agents that run make targets get workspace-write; the rest are read-only."""
    modes = {
        path.stem: tomllib.loads(text)["sandbox_mode"]
        for path, text in sync_agents.build(ROOT).items()
    }
    writers = {"reviewer", "bug-hunter", "perf-auditor", "docs-auditor"}
    assert {name for name, mode in modes.items() if mode != "read-only"} == writers
    assert all(modes[name] == "workspace-write" for name in writers)


def test_bad_sandbox_value_rejected(sync_agents: ModuleType) -> None:
    """An unknown codex-sandbox value is an error, not a silent default."""
    with pytest.raises(sync_agents.AgentFileError, match="codex-sandbox"):
        sync_agents.sandbox_of({"codex-sandbox": "danger-full-access"}, "x.md")


def test_toml_multiline_escapes(sync_agents: ModuleType) -> None:
    """Backslashes and triple quotes survive the TOML round trip."""
    for value in ("a \\| b\n", "x ''' y\n", 'q """ \\n r\n'):
        rendered = f"v = {sync_agents.toml_multiline(value)}\n"
        assert tomllib.loads(rendered)["v"] == value


def test_check_reports_stale_and_orphans(
    sync_agents: ModuleType, tmp_path: Path
) -> None:
    """--check fails on a missing output and on an orphaned generated file."""
    source = tmp_path / ".claude" / "agents"
    source.mkdir(parents=True)
    (source / "demo.md").write_text(
        "---\nname: demo\ndescription: Demo agent.\ntools: Read, Edit\nmodel: inherit\n"
        "---\nDo it.\n\n## Report format\n\nOne line.\n",
        encoding="utf-8",
    )
    (source / "VENDORED.md").write_text("# Vendored\n", encoding="utf-8")
    assert sync_agents.main(["--check"], root=tmp_path) == 1
    assert sync_agents.main([], root=tmp_path) == 0
    target = tmp_path / ".codex" / "agents"
    assert sorted(p.name for p in target.iterdir()) == ["demo.toml"]
    assert "sandbox_mode" not in (target / "demo.toml").read_text(encoding="utf-8")
    assert sync_agents.main(["--check"], root=tmp_path) == 0
    (target / "gone.toml").write_text("# Generated from x\n", encoding="utf-8")
    (target / "manual.toml").write_text('name = "manual"\n', encoding="utf-8")
    assert sync_agents.main(["--check"], root=tmp_path) == 1
    assert sync_agents.main([], root=tmp_path) == 0
    assert sorted(p.name for p in target.iterdir()) == ["demo.toml", "manual.toml"]
