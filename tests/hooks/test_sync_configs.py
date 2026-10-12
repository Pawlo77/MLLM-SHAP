"""The agent tool configs are generated from one source and never drift.

@see [[scripts/sync_configs.py]]
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import HOOKS, ROOT

SHIP = ("Bash(git commit*)", "Bash(git push origin *)", "Bash(gh pr merge*)")


@pytest.fixture(scope="module")
def configs() -> ModuleType:
    """Load scripts/sync_configs.py."""
    spec = importlib.util.spec_from_file_location(
        "sync_configs", ROOT / "scripts" / "sync_configs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _root(tmp_path: Path, table: str = "") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.agent-kit]\nname = "demo"\n' + table, encoding="utf-8"
    )
    return tmp_path


def _scripts(group: object) -> list[str]:
    """Hook script names in a Claude- or Codex-shaped event list, in order."""
    assert isinstance(group, list)
    return [
        str(h["command"]).rstrip('"').rsplit("/", 1)[1].removesuffix('"')
        for g in group
        for h in g["hooks"]
    ]


def test_every_tool_registers_the_shared_gates(
    configs: ModuleType, tmp_path: Path
) -> None:
    root = _root(tmp_path)
    claude = configs.claude_settings(root)["hooks"]
    codex = configs.codex_hooks(root)["hooks"]
    cursor = configs.cursor_hooks(root)["hooks"]
    for scripts in (
        _scripts(claude["PreToolUse"]),
        _scripts(codex["PreToolUse"]),
    ):
        assert scripts == ["shell_guard.sh", "merge_gate.sh"]
    assert _scripts(claude["Stop"]) == ["stop_test_gate.sh"] == _scripts(codex["Stop"])
    assert _scripts(claude["SessionStart"]) == ["session_start.sh"]
    assert [h["command"] for h in cursor["beforeShellExecution"]] == [
        "bash scripts/hooks/shell_guard.sh",
        "bash scripts/hooks/merge_gate.sh",
    ]
    assert all(h["failClosed"] for h in cursor["beforeShellExecution"])
    named = {
        "shell_guard.sh",
        "merge_gate.sh",
        "stop_test_gate.sh",
        "session_start.sh",
        "format_after_edit.sh",
    }
    assert all((HOOKS / name).is_file() for name in named)


def test_codex_events_mirror_claude_events(configs: ModuleType, tmp_path: Path) -> None:
    root = _root(tmp_path, '[tool.agent-kit.configs]\npost-edit-hooks = ["extra.sh"]\n')
    claude = configs.claude_settings(root)["hooks"]
    codex = configs.codex_hooks(root)["hooks"]
    for event, groups in codex.items():
        assert event in claude, event
        assert set(_scripts(groups)) <= set(_scripts(claude[event])), event


def test_delivery_switch_controls_commit_push_and_merge(
    configs: ModuleType, tmp_path: Path
) -> None:
    ask = configs.claude_settings(_root(tmp_path / "ask"))["permissions"]["allow"]
    assert not set(SHIP) & set(ask)
    table = '[tool.agent-kit.configs]\ndelivery = "autopilot"\n'
    auto = configs.claude_settings(_root(tmp_path / "auto", table))
    assert set(SHIP) <= set(auto["permissions"]["allow"])


def test_secrets_and_force_pushes_are_always_denied(
    configs: ModuleType, tmp_path: Path
) -> None:
    table = '[tool.agent-kit.configs]\ndelivery = "autopilot"\n'
    deny = configs.claude_settings(_root(tmp_path, table))["permissions"]["deny"]
    for rule in (
        "Read(**/.env)",
        "Bash(git push --force*)",
        "Bash(git commit --amend*)",
    ):
        assert rule in deny, rule


def test_project_extras_are_appended_without_duplicates(
    configs: ModuleType, tmp_path: Path
) -> None:
    table = (
        "[tool.agent-kit.configs]\n"
        'plugins = ["extra@market", "astral@astral-sh"]\n'
        'allow = ["Bash(make docs*)", "Bash(make check*)"]\n'
        'deny = ["Read(.secret/**)"]\n'
        '[tool.agent-kit.configs.marketplaces]\nmarket = "o/market"\n'
    )
    data = configs.claude_settings(_root(tmp_path, table))
    allow = data["permissions"]["allow"]
    assert allow[-1] == "Bash(make docs*)"
    assert len(allow) == len(set(allow))
    assert data["permissions"]["deny"][-1] == "Read(.secret/**)"
    assert list(data["enabledPlugins"]).count("astral@astral-sh") == 1
    assert "extra@market" in data["enabledPlugins"]
    assert data["extraKnownMarketplaces"]["market"]["source"]["repo"] == "o/market"
    assert "astral-sh" in data["extraKnownMarketplaces"]


def test_post_edit_hooks_reach_all_three_tools(
    configs: ModuleType, tmp_path: Path
) -> None:
    root = _root(tmp_path, '[tool.agent-kit.configs]\npost-edit-hooks = ["extra.sh"]\n')
    claude = configs.claude_settings(root)["hooks"]["PostToolUse"]
    codex = configs.codex_hooks(root)["hooks"]["PostToolUse"]
    cursor = configs.cursor_hooks(root)["hooks"]
    assert _scripts(claude) == ["format_after_edit.sh", "extra.sh"]
    assert _scripts(codex) == ["format_after_edit.sh", "extra.sh"]
    assert cursor["afterFileEdit"][-1]["command"] == "bash scripts/hooks/extra.sh"
    assert len(cursor["afterTabFileEdit"]) == 1


def test_codex_config_lists_the_approval_commands(
    configs: ModuleType, tmp_path: Path
) -> None:
    root = _root(tmp_path, '[tool.agent-kit.configs]\ncodex-approval = ["sbatch"]\n')
    text = configs.codex_config(root)
    assert 'approval_policy = "on-request"' in text
    assert "gh pr merge" in text
    assert "sbatch." in text
    assert all(len(line) <= 88 for line in text.splitlines() if line.startswith("#   "))


def test_check_reports_stale_files_and_write_fixes_them(
    configs: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _root(tmp_path)
    assert configs.main(["--check", "--root", str(root)]) == 1
    assert not (root / ".claude" / "settings.json").exists()  # --check never writes
    assert configs.main(["--root", str(root)]) == 0
    assert configs.main(["--check", "--root", str(root)]) == 0
    json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
    (root / ".cursor" / "hooks.json").write_text("{}\n", encoding="utf-8")
    capsys.readouterr()
    assert configs.main(["--check", "--root", str(root)]) == 1
    assert "stale: .cursor/hooks.json" in capsys.readouterr().out
