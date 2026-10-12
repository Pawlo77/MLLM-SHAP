"""Agent tool configs register the shared hooks and deny reading secrets.

Project-owned: it checks this repository's own ``.claude``, ``.codex`` and ``.cursor``
files (not template-managed yet).

@see [[scripts/hooks/merge_gate.sh]]
"""

import fnmatch
import json
import os

from tests.hooks._hookrun import HOOKS, ROOT

GATES = {
    "session_start.sh",
    "merge_gate.sh",
    "format_after_edit.sh",
    "stop_test_gate.sh",
}


def _commands(hooks: dict[str, list[dict[str, object]]]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for event, groups in hooks.items():
        for group in groups:
            entries = group.get("hooks", [group])
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, dict)
                name = str(entry["command"]).rsplit("/", 1)[-1].strip('"')
                found.setdefault(event, set()).add(name)
    return found


def test_gates_registered_in_all_three_tools() -> None:
    """Claude Code, Codex and Cursor register the kept hooks on their events."""
    claude = _commands(
        json.loads((ROOT / ".claude" / "settings.json").read_text())["hooks"]
    )
    assert "session_start.sh" in claude["SessionStart"]
    assert "merge_gate.sh" in claude["PreToolUse"]
    assert "format_after_edit.sh" in claude["PostToolUse"]
    assert "stop_test_gate.sh" in claude["Stop"]
    codex = _commands(json.loads((ROOT / ".codex" / "hooks.json").read_text())["hooks"])
    assert "session_start.sh" in codex["SessionStart"]
    assert "merge_gate.sh" in codex["PreToolUse"]
    assert "format_after_edit.sh" in codex["PostToolUse"]
    assert "stop_test_gate.sh" in codex["Stop"]
    cursor = _commands(
        json.loads((ROOT / ".cursor" / "hooks.json").read_text())["hooks"]
    )
    assert "session_start.sh" in cursor["sessionStart"]
    assert "merge_gate.sh" in cursor["beforeShellExecution"]
    assert "format_after_edit.sh" in cursor["afterFileEdit"]
    assert "stop_test_gate.sh" in cursor["stop"]
    every = set().union(*claude.values(), *codex.values(), *cursor.values())
    for name in every:
        script = HOOKS / name
        assert script.is_file(), name
        assert os.access(script, os.X_OK), name
    assert every >= GATES


def test_codex_hooks_mirror_claude_events() -> None:
    """Codex uses the Claude event names and the same scripts for shared events."""
    claude = _commands(
        json.loads((ROOT / ".claude" / "settings.json").read_text())["hooks"]
    )
    codex = _commands(json.loads((ROOT / ".codex" / "hooks.json").read_text())["hooks"])
    for event, names in codex.items():
        assert event in claude, event
        assert names <= claude[event], event


def test_claude_read_denies_cover_cursorignore() -> None:
    """Every .cursorignore entry and the key-file names are Read denies for Claude."""
    deny = json.loads((ROOT / ".claude" / "settings.json").read_text())["permissions"][
        "deny"
    ]
    patterns = [
        p[len("Read(") : -1].removeprefix("./").removeprefix("**/")
        for p in deny
        if p.startswith("Read(")
    ]
    entries = [
        line.strip()
        for line in (ROOT / ".cursorignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    keys = [".pypirc", "id_ed25519", "id_rsa", ".netrc", ".git-credentials"]
    # ``.env.*`` in .cursorignore is covered by enumerated secret names so that
    # ``.env.example`` stays readable; check those names instead of the glob.
    secret_env = [".env.local", ".env.x.local", ".env.production", ".env.staging"]
    for entry in [*entries, *keys]:
        name = entry.removeprefix("**/").rstrip("/")
        names = secret_env if name == ".env.*" else [name]
        for candidate in names:
            assert any(
                fnmatch.fnmatch(candidate, pat) or pat == f"{candidate}/**"
                for pat in patterns
            ), entry
    assert not any(fnmatch.fnmatch(".env.example", pat) for pat in patterns)
    assert {"**/.pypirc", "**/.netrc", "**/.git-credentials"} <= {
        p[len("Read(") : -1] for p in deny if p.startswith("Read(")
    }
