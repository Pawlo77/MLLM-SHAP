"""Generate the agent tool configs from one source.

``.claude/settings.json``, ``.codex/hooks.json``, ``.codex/config.toml`` and
``.cursor/hooks.json`` come from a shared base plus ``[tool.agent-kit.configs]``:
``delivery`` (``ask`` or ``autopilot``: may Claude commit, push and merge unasked),
``allow``, ``deny``, ``marketplaces``, ``plugins``, ``post-edit-hooks``,
``codex-approval``. Never edit the outputs.

Usage: ``uv run python scripts/sync_configs.py [--check]``; exits 1 if stale.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit_config import load as load_config

ROOT = Path(__file__).resolve().parents[1]
"""Repository root."""

MARKETPLACES = {"astral-sh": "astral-sh/claude-code-plugins"}
"""Plugin marketplaces every repository declares."""

PLUGINS = (
    "superpowers@claude-plugins-official",
    "code-review@claude-plugins-official",
    "security-guidance@claude-plugins-official",
    "context7@claude-plugins-official",
    "skill-creator@claude-plugins-official",
    "remember@claude-plugins-official",
    "astral@astral-sh",
)
"""Plugins every repository enables."""

ENV = {"ENABLE_STOP_REVIEW": "0", "CLAUDE_CODE_SUBAGENT_MODEL": "sonnet"}
"""Environment of every Claude Code session."""

ALLOW = (
    "Bash(make check*)",
    "Bash(make tests*)",
    "Bash(make prek*)",
    "Bash(uv run pytest*)",
    "Bash(uv run ruff*)",
    "Bash(uv run mypy*)",
    "Bash(uv run prek*)",
    "Bash(uv run python .agents/skills/*)",
    "Bash(uv lock*)",
    "Bash(uv sync*)",
    "Bash(git status*)",
    "Bash(git diff*)",
    "Bash(git log*)",
    "Bash(git branch*)",
    "Bash(git checkout -b *)",
    "Bash(git switch -c *)",
    "Bash(git add *)",
    "Bash(gh pr view*)",
    "Bash(gh pr diff*)",
    "Bash(gh pr checks*)",
    "Bash(git pull --ff-only*)",
    "Bash(gh run list*)",
    "Bash(gh run view*)",
    "Bash(gh run watch*)",
    "WebFetch(domain:api.semanticscholar.org)",
    "WebFetch(domain:export.arxiv.org)",
    "WebFetch(domain:arxiv.org)",
    "WebFetch(domain:doi.org)",
    "WebFetch(domain:api.crossref.org)",
    "Read(**)",
    "Edit(src/**)",
    "Edit(tests/**)",
    "Edit(.agents/skills/**)",
    "Edit(.remember/**)",
    "Edit(plans/**)",
    "Bash(make vault-check*)",
)
"""Permission rules that are safe in every repository."""

AUTOPILOT = (
    "Bash(git commit*)",
    "Bash(git push origin *)",
    "Bash(git push -u origin *)",
    "Bash(gh pr create*)",
    "Bash(gh pr edit*)",
    "Bash(gh pr comment*)",
    "Bash(gh pr merge*)",
    "Bash(gh api *)",
    "Bash(git tag v*)",
    "Bash(git tag -a v*)",
    "Bash(git push origin v*)",
    "Bash(git switch main)",
    "Bash(make upgrade*)",
    "Bash(uv lock --upgrade*)",
    "Bash(uv run prek update*)",
)
"""Added by ``delivery = "autopilot"``: skill ``ship`` commits, pushes and merges."""

DENY = (
    "Read(**/.env)",
    "Read(**/.env.local)",
    "Read(**/.env.*.local)",
    "Read(**/.env.prod*)",
    "Read(**/.env.stag*)",
    "Read(**/.env.dev*)",
    "Read(**/.env.test*)",
    "Read(**/.env.secret*)",
    "Read(**/.pypirc)",
    "Read(**/*.pem)",
    "Read(**/*.key)",
    "Read(**/credentials.json)",
    "Read(**/secrets.json)",
    "Read(**/id_rsa*)",
    "Read(**/id_ed25519*)",
    "Read(**/.netrc)",
    "Read(**/.git-credentials)",
    "Bash(git push --force*)",
    "Bash(git push -f*)",
    "Bash(git push --mirror*)",
    "Bash(git push origin main*)",
    "Bash(git push origin master*)",
    "Bash(git push origin HEAD:main*)",
    "Bash(git commit --no-verify*)",
    "Bash(git commit -n *)",
    "Bash(git commit --amend*)",
    "Bash(gh pr create --draft*)",
    "Bash(gh pr merge * --squash*)",
    "Bash(gh pr merge * --rebase*)",
    "Bash(gh pr merge * --admin*)",
    "Bash(pip install*)",
    "Bash(pip3 install*)",
    "Bash(pre-commit*)",
    "Bash(cat *.env*)",
)
"""Reads and commands that are never allowed."""

CODEX_APPROVAL = (
    "git commit",
    "git push",
    "git tag",
    "gh pr create",
    "gh pr merge",
    "gh release create",
    "uv publish",
    "any force push",
    "any push to main",
)
"""Commands Codex must ask about, whatever the approval mode."""

POST_EDIT_TIMEOUT = 10
"""Seconds an extra post-edit hook may run."""


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _table(root: Path) -> dict[str, Any]:
    raw = load_config(root).get("configs", {})
    return raw if isinstance(raw, dict) else {}


def _claude(script: str) -> str:
    return f'"${{CLAUDE_PROJECT_DIR:-.}}"/scripts/hooks/{script}'


def _codex(script: str) -> str:
    return f'bash "$(git rev-parse --show-toplevel)/scripts/hooks/{script}"'


def _command(cmd: str, timeout: int, status: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"type": "command", "command": cmd, "timeout": timeout}
    if status:
        entry["statusMessage"] = status
    return entry


def claude_settings(root: Path = ROOT) -> dict[str, Any]:
    """Return the content of ``.claude/settings.json``."""
    cfg = _table(root)
    autopilot = cfg.get("delivery", "ask") == "autopilot"
    marketplaces = {**MARKETPLACES, **cfg.get("marketplaces", {})}
    post = [_command(_claude("format_after_edit.sh"), 30, "Formatting with Ruff")]
    post += [
        _command(_claude(n), POST_EDIT_TIMEOUT) for n in cfg.get("post-edit-hooks", [])
    ]
    allow = [*ALLOW, *(AUTOPILOT if autopilot else ()), *cfg.get("allow", [])]
    return {
        "extraKnownMarketplaces": {
            name: {"source": {"source": "github", "repo": repo}}
            for name, repo in marketplaces.items()
        },
        "enabledPlugins": dict.fromkeys(
            _unique([*PLUGINS, *cfg.get("plugins", [])]), True
        ),
        "env": ENV,
        "hooks": {
            "SessionStart": [{"hooks": [_command(_claude("session_start.sh"), 90)]}],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        _command(_claude("shell_guard.sh"), 10),
                        _command(_claude("merge_gate.sh"), 30),
                    ],
                }
            ],
            "PostToolUse": [
                {"matcher": "Edit|Write|MultiEdit|NotebookEdit", "hooks": post}
            ],
            "Stop": [
                {
                    "hooks": [
                        _command(
                            _claude("stop_test_gate.sh"),
                            200,
                            "Running make tests before hand-off",
                        )
                    ]
                }
            ],
        },
        "permissions": {
            "allow": _unique(allow),
            "deny": _unique([*DENY, *cfg.get("deny", [])]),
        },
        "statusLine": {"type": "command", "command": _claude("statusline.sh")},
    }


def codex_hooks(root: Path = ROOT) -> dict[str, Any]:
    """Return the content of ``.codex/hooks.json``."""
    cfg = _table(root)
    name = load_config(root).get("name", root.name)
    post = [_command(_codex("format_after_edit.sh"), 30, "Formatting with Ruff")]
    post += [
        _command(_codex(n), POST_EDIT_TIMEOUT) for n in cfg.get("post-edit-hooks", [])
    ]
    return {
        "description": (
            f"{name} project hooks for Codex. Same scripts and Claude-shaped JSON as "
            ".claude/settings.json; Codex has no TaskCompleted, PostToolUseFailure or "
            "FileChanged events."
        ),
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume|clear|compact",
                    "hooks": [_command(_codex("session_start.sh"), 90)],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        _command(_codex("shell_guard.sh"), 10),
                        _command(_codex("merge_gate.sh"), 30),
                    ],
                }
            ],
            "PostToolUse": [{"matcher": "apply_patch|Edit|Write", "hooks": post}],
            "Stop": [
                {
                    "hooks": [
                        _command(
                            _codex("stop_test_gate.sh"),
                            200,
                            "Running make tests before hand-off",
                        )
                    ]
                }
            ],
        },
    }


def cursor_hooks(root: Path = ROOT) -> dict[str, Any]:
    """Return the content of ``.cursor/hooks.json``."""
    extra = [
        {"command": f"bash scripts/hooks/{n}"}
        for n in _table(root).get("post-edit-hooks", [])
    ]
    format_hook = {"command": "bash scripts/hooks/format_after_edit.sh"}
    return {
        "version": 1,
        "hooks": {
            "sessionStart": [{"command": "bash scripts/hooks/session_start.sh"}],
            "beforeShellExecution": [
                {"command": "bash scripts/hooks/shell_guard.sh", "failClosed": True},
                {"command": "bash scripts/hooks/merge_gate.sh", "failClosed": True},
            ],
            "afterFileEdit": [format_hook, *extra],
            "afterTabFileEdit": [format_hook],
            "stop": [
                {
                    "command": "bash scripts/hooks/stop_test_gate.sh",
                    "loop_limit": 2,
                    "timeout": 200,
                }
            ],
        },
    }


def codex_config(root: Path = ROOT) -> str:
    """Return the content of ``.codex/config.toml``."""
    extra = list(_table(root).get("codex-approval", []))
    wrapped = _wrap([*CODEX_APPROVAL, *extra], "#   ")
    return (
        "# Codex CLI project configuration "
        "(https://developers.openai.com/codex/config-reference).\n"
        "# Codex reads AGENTS.md and .agents/skills/ natively. Project hooks live in "
        ".codex/hooks.json\n"
        "# (the shared scripts/hooks/*.sh; trust them once in Codex's hook review) and "
        "subagents in\n"
        "# .codex/agents/*.toml, generated from .claude/agents/*.md by "
        "scripts/sync_agents.py.\n\n"
        "# Edits stay inside the workspace; network and paths outside it need "
        "approval.\n"
        'sandbox_mode = "workspace-write"\n\n'
        "# The agent asks before anything outside the sandbox. These commands must "
        "always "
        "be asked\n"
        "# for explicitly, whatever the approval mode (AGENTS.md, "
        ".cursor/rules/git-commits.mdc):\n"
        f"{wrapped}\n"
        'approval_policy = "on-request"\n'
    )


def _wrap(items: list[str], prefix: str, width: int = 88) -> str:
    """Comma-join ``items`` into comment lines that never split an item."""
    lines, line = [], prefix
    for index, item in enumerate(items):
        piece = item + ("," if index < len(items) - 1 else ".")
        if line != prefix and len(line) + 1 + len(piece) > width:
            lines.append(line)
            line = prefix
        line += ("" if line == prefix else " ") + piece
    lines.append(line)
    return "\n".join(lines)


def render(root: Path = ROOT) -> dict[str, str]:
    """Return every generated file, keyed by its path relative to ``root``."""

    def dump(data: dict[str, Any]) -> str:
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    return {
        ".claude/settings.json": dump(claude_settings(root)),
        ".codex/hooks.json": dump(codex_hooks(root)),
        ".codex/config.toml": codex_config(root),
        ".cursor/hooks.json": dump(cursor_hooks(root)),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the configs, or with ``--check`` report the stale ones."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when stale")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    stale = []
    for rel, content in render(args.root).items():
        path = args.root / rel
        if path.is_file() and path.read_text(encoding="utf-8") == content:
            continue
        stale.append(rel)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    for rel in stale:
        print(f"{'stale' if args.check else 'wrote'}: {rel}")
    if args.check and stale:
        print("run: uv run python scripts/sync_configs.py", file=sys.stderr)
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
