"""Classify an agent shell command as allow, ask or deny (stdlib only).

Usage: ``python3 scripts/hooks/shell_guard.py "<command>"`` prints one JSON object
``{"permission": ..., "reason": ...}``; ``shell_guard.sh`` wraps it for Claude Code,
Codex and Cursor. Denies skipped hooks, force pushes, amends, malformed commit messages
and draft or WIP pull requests. Asks before cluster jobs (``sbatch``, ``scancel``, also
over ``ssh``) and ``rsync``/``scp`` uploads.
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
ASK_SCRIPTS: tuple[str, ...] = ()


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
    patterns = []
    for pattern in cfg.get("ask-patterns", []):
        try:
            patterns.append(re.compile(pattern))
        except re.error:
            continue  # a bad project pattern must not disable the built-in rules
    return patterns, tuple(cfg.get("ask-scripts", ASK_SCRIPTS))


INTERPRETER = re.compile(r"\b(bash|sh|zsh|dash|python3?|node|perl|ruby|eval|source)\b")
WRAPPER = re.compile(
    r"^(?:[({]\s*|(?:env|command|time|nice|sudo|exec)\s+(?:-\S+\s+)*)+"
)
SUBSTITUTION = re.compile(r"\$\(([^()]*)\)|`([^`]*)`")
SHELL_C = re.compile(r"^(?:bash|sh|zsh|dash)\s+(?:-\w+\s+)*-\w*c\w*\s+")
VALUE_FLAGS = "mFCct"  # short options that take a value, ending a flag cluster
GIT_CONFIG_ENV = re.compile(
    r"\bGIT_CONFIG(_GLOBAL|_SYSTEM|_COUNT|_KEY_\d+|_VALUE_\d+|_PARAMETERS)?=",
    re.IGNORECASE,
)
HOOK_CONFIG = ("hookspath", "include.path", "includeif")
GIT_GLOBAL_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


def _heredoc(match: re.Match[str]) -> str:
    """Drop a heredoc body, but keep it when it feeds an interpreter."""
    text = match.string
    line = text[text.rfind("\n", 0, match.start()) + 1 : match.start()]
    return f"\n{match.group(2)}\n" if INTERPRETER.search(line) else ""


def segments(command: str) -> list[str]:
    """Return top-level commands without data heredocs, wrappers and env assignments."""
    flat = HEREDOC.sub(_heredoc, command)
    result = []
    for raw in SEGMENT.split(flat):
        part = ENV_ASSIGN.sub("", raw.strip())
        grouped = part.startswith(("(", "{"))
        part = ENV_ASSIGN.sub("", WRAPPER.sub("", part))
        part = part.rstrip(" )}") if grouped else part
        if part:
            result.append(part)
    return result


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


def _git_split(words: list[str]) -> tuple[list[str], str, list[str]]:
    """Split ``git [global options] <subcommand> [args]`` into its three parts."""
    index, global_opts = 1, []
    while index < len(words) and words[index].startswith("-"):
        global_opts.append(words[index])
        if words[index] in GIT_GLOBAL_WITH_VALUE and index + 1 < len(words):
            index += 1
            global_opts.append(words[index])
        index += 1
    sub = words[index] if index < len(words) else ""
    return global_opts, sub, words[index + 1 :]


def _short_flags(args: list[str]) -> set[str]:
    """Return the single-letter flags before any value-taking flag of each cluster."""
    flags: set[str] = set()
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif re.fullmatch(r"-[A-Za-z]+", arg):
            for char in arg[1:]:
                flags.add(char)
                if char in VALUE_FLAGS:
                    skip = arg.endswith(char)
                    break
        elif arg in {"--message", "--file", "--author", "--date"}:
            skip = True
    return flags


def _long(args: list[str], option: str) -> bool:
    """Return whether ``option`` is given, also as git's unambiguous abbreviation."""
    for arg in args:
        name = arg.split("=", 1)[0]
        if name.startswith("--") and len(name) >= 4 and option.startswith(name):
            return True
    return False


def _git_decision(part: str, command: str) -> tuple[str, str] | None:
    try:
        words = shlex.split(part)
    except ValueError:
        if "<<" not in command:
            return "ask", "Cannot parse this git command; confirm it is intended."
        words = part.split()  # a heredoc split the quoted message across segments
    global_opts, sub, args = _git_split(words)
    config_words = global_opts + (args if sub == "config" else [])
    hooks_override = any(key in w.lower() for w in config_words for key in HOOK_CONFIG)
    if hooks_override or GIT_CONFIG_ENV.search(command):
        return "deny", "Overriding git hooks through config is not allowed."
    if _long(args, "--no-verify") or (sub == "commit" and "n" in _short_flags(args)):
        return "deny", "--no-verify is not allowed; fix the failing hook instead."
    if sub == "push" and (
        any(_long(args, opt) for opt in ("--force", "--mirror", "--force-with-lease"))
        or "f" in _short_flags(args)
        or any(arg.startswith("+") for arg in args)
    ):
        return "deny", "Force pushes are not allowed; push a new commit instead."
    if sub == "commit":
        if _long(args, "--amend"):
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


def _nested(command: str, part: str, depth: int) -> tuple[str, str] | None:
    """Classify ``$(...)``, backtick and ``bash -c`` payloads; first non-allow wins."""
    payloads = [a or b for a, b in SUBSTITUTION.findall(command)]
    if SHELL_C.match(part):
        try:
            payloads.append(shlex.split(part)[-1])
        except ValueError:
            return "ask", "Cannot parse this shell payload; confirm it is intended."
    for payload in payloads:
        result = classify(payload, depth + 1)
        if result[0] != "allow":
            return result
    return None


def classify(command: str, depth: int = 0) -> tuple[str, str]:
    """Return ``(permission, reason)`` for one shell command line."""
    asks: list[str] = []
    patterns, scripts = _project_rules()
    if depth > 3:
        return "ask", "Shell nesting is too deep to inspect; confirm it is intended."
    for part in segments(command):
        decision = _git_decision(part, command) if re.match(r"git\b", part) else None
        decision = decision or _pr_decision(part) or _nested(part, part, depth)
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
