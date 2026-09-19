"""Deny risky shell commands and enforce git/PR policy for Cursor agents."""

import json
import re
import sys
from pathlib import Path

_HOOKS = Path(__file__).resolve().parent
if str(_HOOKS) not in sys.path:
    sys.path.insert(0, str(_HOOKS))

from git_policy import decide_git  # noqa: E402


def _bare_command(cmd: str) -> str:
    """Strip quoted spans so echo/test payloads do not false-positive."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", cmd)


def decide_packages(cmd: str) -> dict[str, str] | None:
    """Deny pip/conda/prek-cli footguns; return None if not applicable."""
    bare = _bare_command(cmd)
    if re.search(
        r"(^|[\n;&|])\s*(sudo\s+)?(pip3?|python3?\s+-m\s+pip)\s+install\b",
        bare,
    ):
        return {
            "permission": "deny",
            "user_message": (
                "Use pyproject.toml + uv lock / make install — not pip install."
            ),
            "agent_message": (
                "Denied pip install. Edit pyproject.toml, then "
                "uv lock && uv sync --all-groups (or make install)."
            ),
        }
    if re.search(
        r"(^|[\n;&|])\s*(sudo\s+)?(conda|mamba|pipx)\s+install\b",
        bare,
    ):
        return {
            "permission": "deny",
            "user_message": (
                "Use pyproject.toml + uv lock / make install — not conda/mamba/pipx."
            ),
            "agent_message": (
                "Denied conda/mamba/pipx install. Edit pyproject.toml, then "
                "uv lock && uv sync --all-groups (or make install)."
            ),
        }
    if re.search(r"(^|[\n;&|])\s*(sudo\s+)?pre-commit\b", bare):
        return {
            "permission": "deny",
            "user_message": "Use prek, not pre-commit.",
            "agent_message": (
                "Denied pre-commit CLI. Use: make prek-all or "
                "uv run prek run --all-files."
            ),
        }
    if re.search(
        r"(^|[\n;&|])\s*git\s+push\s+.*(--force\b|--force-with-lease\b|-f\b)",
        bare,
    ):
        return {
            "permission": "deny",
            "user_message": "Force push is blocked by project policy.",
            "agent_message": (
                "Denied git force-push. Ask the user explicitly; never force-push main."
            ),
        }
    return None


def decide(cmd: str) -> dict[str, str]:
    """Return the Cursor permission payload for ``cmd``."""
    for checker in (decide_packages, decide_git):
        result = checker(cmd)
        if result is not None:
            return result
    return {"permission": "allow"}


def main() -> None:
    """Read hook JSON from stdin and print the permission decision."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    cmd = payload.get("command") or ""
    print(json.dumps(decide(cmd if isinstance(cmd, str) else "")))


if __name__ == "__main__":
    main()
