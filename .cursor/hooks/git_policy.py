"""Cursor beforeShellExecution policy for git commits, pushes, and PRs."""

import json
import re
import subprocess
import sys
from pathlib import Path

_HOOKS = Path(__file__).resolve().parent
if str(_HOOKS) not in sys.path:
    sys.path.insert(0, str(_HOOKS))

from commit_msg import validate_commit_message  # noqa: E402

_COMMIT = re.compile(r"(^|[\n;&|])\s*git\s+commit\b")
_PUSH = re.compile(r"(^|[\n;&|])\s*git\s+push\b")
_PR = re.compile(r"(^|[\n;&|])\s*gh\s+pr\s+(create|merge|ready|edit)\b")
_NO_VERIFY = re.compile(r"(^|[\s])(--no-verify|-n)(?:=|\s|$)")
_MSG_FLAG = re.compile(r"""(?:^|\s)(?:-m|--message)(?:\s+|=)(?P<q>'[^']*'|"[^"]*")""")
_HEREDOC = re.compile(
    r"""(?:-m|--message)\s+["']?\$\(cat\s+<<['"]?(?P<tag>\w+)['"]?\n(?P<body>.*?)^(?P=tag)\s*\)""",
    re.MULTILINE | re.DOTALL,
)
_FILE_FLAG = re.compile(r"(?:^|\s)(?:-F|--file)(?:\s+|=)\S+")
_EXPLICIT_MAIN = re.compile(
    r"(?:^|\s)(?:refs/heads/)?(main|master)(?:\s|$)|"
    r"HEAD:(?:refs/heads/)?(main|master)\b"
)


def _bare(cmd: str) -> str:
    """Strip quoted spans for coarse matching."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", cmd)


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def extract_commit_message(cmd: str) -> str | None:
    """Extract commit message text from a ``git commit`` shell command."""
    heredoc = _HEREDOC.search(cmd)
    if heredoc:
        return heredoc.group("body")
    parts = [_unquote(m.group("q")) for m in _MSG_FLAG.finditer(cmd)]
    if parts:
        return "\n\n".join(parts)
    return None


def _current_branch() -> str:
    """Return the current git branch name, or empty string on failure."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _push_targets_main(cmd: str, bare: str) -> bool:
    """Return True if the push command targets main/master."""
    if _EXPLICIT_MAIN.search(bare) and re.search(r"git\s+push\b", bare):
        # Avoid false positive on branch names inside unrelated flags; require
        # main/master as a push ref (after origin or as sole ref).
        if re.search(
            r"git\s+push\b(?:\s+\S+)*\s+(?:origin\s+)?(?:HEAD:)?(?:refs/heads/)?"
            r"(main|master)\b",
            bare,
        ):
            return True
        if re.search(r"HEAD:(?:refs/heads/)?(main|master)\b", cmd):
            return True
    # Bare push / push HEAD while checked out on main
    if re.search(
        r"git\s+push(?:\s+(-u|--set-upstream))?(?:\s+origin)?(?:\s+HEAD)?\s*$",
        bare.strip(),
    ) or re.search(r"git\s+push\s+(-u|--set-upstream)\s+\S+\s+HEAD\s*$", bare):
        return _current_branch() in {"main", "master"}
    return False


def decide_git(cmd: str) -> dict[str, str] | None:
    """Return a permission payload for git/gh policy, or ``None`` if N/A."""
    bare = _bare(cmd)

    if _COMMIT.search(bare):
        if _NO_VERIFY.search(bare):
            return {
                "permission": "deny",
                "user_message": "git commit --no-verify is blocked.",
                "agent_message": (
                    "Denied --no-verify. Hooks must run (prek commit-msg + pre-commit)."
                ),
            }
        has_msg = bool(_MSG_FLAG.search(cmd) or _HEREDOC.search(cmd))
        has_file = bool(_FILE_FLAG.search(cmd))
        if not has_msg and not has_file:
            return {
                "permission": "deny",
                "user_message": "git commit must use -m/-F (no interactive editor).",
                "agent_message": (
                    "Denied interactive git commit. Use HEREDOC -m per git-commits.mdc."
                ),
            }
        if has_msg:
            message = extract_commit_message(cmd)
            if message is not None:
                err = validate_commit_message(message)
                if err:
                    return {
                        "permission": "deny",
                        "user_message": f"Invalid commit message: {err}",
                        "agent_message": (
                            f"Denied commit: {err}. "
                            "Use type: summary "
                            "(feat|fix|docs|test|refactor|chore|ci|build|perf); "
                            "no trailing period; ≤72 chars."
                        ),
                    }
        # Valid commit shape — allow without Cursor approval prompt.
        # User permission still required by git-commits.mdc / AGENTS.md.
        return None

    if _PUSH.search(bare):
        if _push_targets_main(cmd, bare):
            return {
                "permission": "deny",
                "user_message": "Direct push to main/master is blocked.",
                "agent_message": (
                    "Denied push to main/master. Push a feature branch and open a PR "
                    "(gh pr create) after user approval."
                ),
            }
        # Feature-branch push — allow; after-push hook reminds about CI watch.
        return None

    if _PR.search(bare):
        # Allow without Cursor approval prompt; AGENTS.md still requires user ask.
        return None

    return None


def main() -> None:
    """Read hook JSON from stdin; print permission decision for tests."""
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    cmd = payload.get("command") or ""
    result = decide_git(cmd if isinstance(cmd, str) else "")
    print(json.dumps(result or {"permission": "allow"}))


if __name__ == "__main__":
    main()
