"""The Claude Code config registers the shared hooks and denies reading secrets.

Project-owned: it checks this repository's ``.claude/settings.json`` (Claude Code is the
only agent tool, ``agent_tools: [claude]``).

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


def test_gates_registered_in_claude_code() -> None:
    """Claude Code registers the kept hooks on their events."""
    claude = _commands(
        json.loads((ROOT / ".claude" / "settings.json").read_text())["hooks"]
    )
    assert "session_start.sh" in claude["SessionStart"]
    assert "merge_gate.sh" in claude["PreToolUse"]
    assert "format_after_edit.sh" in claude["PostToolUse"]
    assert "stop_test_gate.sh" in claude["Stop"]
    every = set().union(*claude.values())
    for name in every:
        script = HOOKS / name
        assert script.is_file(), name
        assert os.access(script, os.X_OK), name
    assert every >= GATES


def test_claude_read_denies_cover_secrets() -> None:
    """The secret and key-file names are Read denies for Claude."""
    deny = json.loads((ROOT / ".claude" / "settings.json").read_text())["permissions"][
        "deny"
    ]
    patterns = [
        p[len("Read(") : -1].removeprefix("./").removeprefix("**/")
        for p in deny
        if p.startswith("Read(")
    ]
    entries = [  # the former .cursorignore
        ".env",
        ".env.*",
        ".pypirc",
        "*.pem",
        "*.key",
        "**/credentials.json",
        "**/secrets.json",
        "**/id_rsa",
        "**/id_rsa.pub",
        "**/id_ed25519*",
        ".netrc",
        ".git-credentials",
        ".smart-env/",
        ".obsidian/plugins/realclaudian/data.json",
        ".claude/jit-context/.discovery/",
    ]
    keys = [".pypirc", "id_ed25519", "id_rsa", ".netrc", ".git-credentials"]
    # ``.env.*`` is covered by enumerated secret names so that
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
