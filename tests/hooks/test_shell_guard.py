"""Shell guard: deny, ask and allow decisions for agent shell commands.

@see [[scripts/hooks/shell_guard.py]]
"""

import importlib.util
import json
import re
from types import ModuleType
from typing import Any

import pytest

from tests.hooks._hookrun import CLAUDE, CURSOR, HAVE_TOOLS, HOOKS, run_hook

ASK_PATTERNS = (
    r"^(uv\s+publish|twine\s+upload)\b",
    r"^gh\s+release\s+create\b",
    r"^gh\s+api\b.*(-X|--method)[\s=]+DELETE\b",
)


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    """Load scripts/hooks/shell_guard.py."""
    spec = importlib.util.spec_from_file_location(
        "shell_guard", HOOKS / "shell_guard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Pin the project rules, so the cases do not depend on the repository's config.
    module._project_rules = lambda: (
        [re.compile(p) for p in ASK_PATTERNS],
        ("deploy.sh",),
    )
    return module


HEREDOC_COMMIT = """git commit -m "$(cat <<'EOF'
feat: from heredoc

Body line
EOF
)" """

HEREDOC_BAD = "git commit -q -F - <<'EOF'\nwip stuff\nEOF"

PR_BODY = (
    "gh pr create --title 'feat: x' --body \"$(cat <<'EOF'\n"
    "mentions git commit -m 'bad' and sbatch inside the PR body\n"
    'EOF\n)"\n'
)


@pytest.mark.parametrize(
    ("command", "permission"),
    [
        ("make check", "allow"),
        ("git commit -m 'feat: add hook'", "allow"),
        ('git commit -m "fix(tv): resume"', "allow"),
        (HEREDOC_COMMIT, "allow"),
        ("gh pr create --title 'feat: x' --body 'y'", "allow"),
        (PR_BODY, "allow"),
        ("ssh helios squeue -u me", "allow"),
        ("rsync -a helios:/scratch/run ./local", "allow"),
        ("git commit --no-verify -m 'feat: x'", "deny"),
        ("git push --force origin HEAD", "deny"),
        ("git push -f", "deny"),
        ("git -C . push --force origin HEAD", "deny"),
        ("git -c user.name=x push -fu origin HEAD", "deny"),
        ("command git push -f", "deny"),
        ("env GIT_TRACE=1 git push --force-with-lease", "deny"),
        ("(git push -f)", "deny"),
        ("echo $(git push -f)", "deny"),
        ("bash -c 'git push --force'", "deny"),
        ("bash <<'EOF'\ngit push --force\nEOF", "deny"),
        ("git commit --no-ver -m 'feat: x'", "deny"),
        ("git push --forc origin HEAD", "deny"),
        ("git commit --amen --no-edit", "deny"),
        ("git config core.hooksPath /dev/null", "deny"),
        (
            "GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.hooksPath git commit -m 'x'",
            "deny",
        ),
        ("git commit -n -m 'feat: x'", "deny"),
        ("git commit -an -m 'feat: x'", "deny"),
        ("git -c core.hooksPath=/dev/null commit -m 'feat: x'", "deny"),
        ("git commit -m 'feat: none of the n flags'", "allow"),
        ("git commit -m'feat: nonsense'", "allow"),
        ("git push -u origin HEAD", "allow"),
        ("git -C sub status", "allow"),
        ("git push 'unterminated", "ask"),
        ("git push origin +main", "deny"),
        ("git commit --amend --no-edit", "deny"),
        ("git commit -m 'bad message'", "deny"),
        (
            "git commit -m 'feat: x' -m 'Co-Authored-By: Claude <c@anthropic.com>'",
            "deny",
        ),
        (HEREDOC_BAD, "deny"),
        ("gh pr create --draft --title 'feat: x'", "deny"),
        ("gh pr create --title 'WIP: feat: x' --body 'y'", "deny"),
        ("sbatch scripts/cluster/helios/run_suite.sbatch", "ask"),
        ("ssh helios 'scancel 123'", "ask"),
        ("bash scripts/deploy.sh", "ask"),
        ("uv publish dist/*", "ask"),
        ("gh release create v1.0.0", "ask"),
        ("gh api -X DELETE repos/Pawlo77/x/git/refs/heads/y", "ask"),
        ("gh api repos/Pawlo77/x/pulls/1", "allow"),
        ("rsync -a outputs/ helios:/scratch/out", "ask"),
        ("scp a.txt helios:/tmp/", "ask"),
    ],
)
def test_classify(guard: ModuleType, command: str, permission: str) -> None:
    assert guard.classify(command)[0] == permission


def test_deny_reason_names_the_rule(guard: ModuleType) -> None:
    assert "type: summary" in guard.classify("git commit -m 'bad message'")[1]
    assert "force" in guard.classify("git push -f")[1].lower()


@pytest.mark.skipif(not HAVE_TOOLS, reason="needs bash and jq")
def test_wrapper_shapes() -> None:
    claude = run_hook(
        HOOKS / "shell_guard.sh", {**CLAUDE, "tool_input": {"command": "sbatch x"}}
    )
    out = json.loads(claude.stdout)["hookSpecificOutput"]
    assert out["permissionDecision"] == "ask"
    cursor = run_hook(HOOKS / "shell_guard.sh", {**CURSOR, "command": "git push -f"})
    assert json.loads(cursor.stdout)["permission"] == "deny"
    allowed = run_hook(
        HOOKS / "shell_guard.sh", {**CLAUDE, "tool_input": {"command": "ls"}}
    )
    assert allowed.stdout == ""
    cursor_allowed = run_hook(HOOKS / "shell_guard.sh", {**CURSOR, "command": "ls"})
    assert json.loads(cursor_allowed.stdout) == {}
