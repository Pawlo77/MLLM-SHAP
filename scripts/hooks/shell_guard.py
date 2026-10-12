"""Classify an agent shell command as allow, ask or deny (stdlib only).

Usage: ``python3 scripts/hooks/shell_guard.py "<command>"`` prints one JSON object
``{"permission": ..., "reason": ...}``; ``shell_guard.sh`` wraps it for Claude Code,
Codex and Cursor. Denies skipped hooks, force pushes, amends, malformed commit messages
and draft or WIP pull requests. Asks before cluster jobs (``sbatch``, ``scancel``, also
over ``ssh``), the submit and deploy scripts, publishing, and ``rsync``/``scp`` uploads.
Projects extend the "ask" rules in ``pyproject.toml``, table
``[tool.agent-kit.shell-guard]``: ``ask-patterns`` (regexes) and ``ask-scripts``.
"""

import json
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commit_msg import validate_commit_message

SEGMENT = re.compile(r"\s*(?:&&|\|\||;|\||\n)\s*")
ENV_ASSIGN = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=(?:\S+|\"[^\"]*\"|'[^']*')\s+)+")
MINUS_M = re.compile(r"""(?:^|\s)-m\s+(?:'([^']*)'|"([^"]*)"|\$'((?:\\'|[^'])*)')""")
HEREDOC = re.compile(r"""<<-?\s*['"]?(\w+)['"]?\n(.*?)\n\1\b""", re.DOTALL)
CLUSTER_JOB = re.compile(r"(^|[\s;&|'\"(])(sbatch|scancel)\b")
ASK_SCRIPTS = ("submit_all.sh", "publish_results.py")


def _project_rules() -> tuple[list[re.Pattern[str]], tuple[str, ...]]:
    """Return extra ask patterns and scripts from ``[tool.agent-kit.shell-guard]``."""
    try:
        import tomllib

        path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))["tool"]["agent-kit"][
            "shell-guard"
        ]
    except (ImportError, OSError, KeyError, ValueError):
        return [], ASK_SCRIPTS
    patterns = [re.compile(p) for p in cfg.get("ask-patterns", [])]
    return patterns, tuple(cfg.get("ask-scripts", ASK_SCRIPTS))


def segments(command: str) -> list[str]:
    """Return top-level commands without heredoc bodies and env assignments."""
    flat = HEREDOC.sub("", command)
    parts = (ENV_ASSIGN.sub("", part.strip()) for part in SEGMENT.split(flat))
    return [part for part in parts if part]


def commit_message(command: str) -> str | None:
    """Return the message of ``git commit -m``, ``-m "$(cat <<EOF)"`` or ``-F -``."""
    heredoc = HEREDOC.search(command)
    from_heredoc = r"git\s+commit\b[^\n]*\s(-F\s+-|-m\s+\"\$\(cat)"
    if heredoc and re.search(from_heredoc, command):
        return heredoc.group(2)
    parts = [
        next(group for group in match.groups() if group is not None)
        for match in MINUS_M.finditer(command)
    ]
    return "\n\n".join(parts) if parts else None


def _git_decision(part: str, command: str) -> tuple[str, str] | None:
    if re.search(r"(^|\s)--no-verify(\s|$)", part):
        return "deny", "--no-verify is not allowed; fix the failing hook instead."
    if re.match(r"git\s+push\b", part) and re.search(
        r"\s(--force|-f|--force-with-lease)(\s|=|$)|\s\+\S", part
    ):
        return "deny", "Force pushes are not allowed; push a new commit instead."
    if re.match(r"git\s+commit\b", part):
        if re.search(r"\s--amend(\s|$)", part):
            return "deny", "git commit --amend is not allowed; create a new commit."
        message = commit_message(command)
        error = validate_commit_message(message) if message is not None else None
        if error:
            return "deny", f"Commit message: {error}"
    return None


def _pr_decision(part: str) -> tuple[str, str] | None:
    if not re.match(r"gh\s+pr\s+(create|edit)\b", part):
        return None
    if re.search(r"(^|\s)--draft(\s|$)", part):
        return "deny", "Draft pull requests are not allowed; open it ready for review."
    if re.search(r"--title(\s+|=)['\"]?\s*\[?WIP\b", part, re.IGNORECASE):
        return "deny", "WIP pull request titles are not allowed."
    return None


def _uploads(part: str) -> bool:
    try:
        words = shlex.split(part)
    except ValueError:
        return False
    if not words or words[0] not in {"rsync", "scp"}:
        return False
    targets = [word for word in words[1:] if not word.startswith("-")]
    return bool(targets) and ":" in targets[-1]


def classify(command: str) -> tuple[str, str]:
    """Return ``(permission, reason)`` for one shell command line."""
    asks: list[str] = []
    patterns, scripts = _project_rules()
    for part in segments(command):
        decision = _git_decision(part, command) if part.startswith("git") else None
        decision = decision or _pr_decision(part)
        if decision:
            return decision
        if CLUSTER_JOB.search(part):
            asks.append("submits or cancels cluster jobs")
        elif any(script in part for script in scripts) or any(
            pattern.search(part) for pattern in patterns
        ):
            asks.append("submits, deploys or publishes")
        elif _uploads(part):
            asks.append("uploads files to a remote host")
    if asks:
        return "ask", f"This command {asks[0]}; confirm it is intended."
    return "allow", ""


def main(argv: list[str] | None = None) -> int:
    """Print the decision for the command in ``argv[0]`` as JSON."""
    args = sys.argv[1:] if argv is None else argv
    permission, reason = classify(args[0] if args else "")
    print(json.dumps({"permission": permission, "reason": reason}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
