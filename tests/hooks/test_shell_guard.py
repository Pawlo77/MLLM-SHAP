"""Shell guard: deny, ask and allow decisions for agent shell commands.

@see [[scripts/hooks/shell_guard.py]]
"""

import importlib.util
import json
from types import ModuleType

import pytest

from tests.hooks._hookrun import CLAUDE, CURSOR, HAVE_TOOLS, HOOKS, run_hook


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    """Load scripts/hooks/shell_guard.py."""
    spec = importlib.util.spec_from_file_location(
        "shell_guard", HOOKS / "shell_guard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
