"""Hook gates: the Stop test gate, session-start state and hook registration.

Every script is fed Claude-shaped (also Codex's shape) and Cursor-shaped payloads.

@see [[scripts/hooks/stop_test_gate.sh]]
@see [[scripts/hooks/session_start.sh]]
"""

import fnmatch
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tests.hooks._hookrun import CLAUDE, CURSOR, HAVE_TOOLS, HOOKS, git, scratch_repo

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="needs bash and jq")

ROOT = HOOKS.parents[1]
TOOLS = (
    "bash",
    "git",
    "jq",
    "date",
    "tr",
    "sort",
    "stat",
    "mkdir",
    "cat",
    "grep",
    "head",
    "tail",
    "awk",
    "sed",
    "mktemp",
    "rm",
    "touch",
    "dirname",
    "cut",
    "basename",
    "pgrep",
    "sleep",
    "env",
)


def run(
    script: str,
    payload: dict[str, object],
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``scripts/hooks/<script>`` of the scratch repo ``cwd`` with ``payload``."""
    env = {**os.environ, **(extra_env or {})}
    env.pop("NLP_SHAP_TEST_GATE", None)
    if extra_env and "NLP_SHAP_TEST_GATE" in extra_env:
        env["NLP_SHAP_TEST_GATE"] = extra_env["NLP_SHAP_TEST_GATE"]
    return subprocess.run(
        ["bash", str(cwd / "scripts" / "hooks" / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=60,
        check=False,
    )


def fake_make(tmp: Path, *, exit_code: int, output: str) -> tuple[Path, Path]:
    """Return a PATH directory holding a fake ``make`` and its call log."""
    bin_dir = tmp / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp / "make.log"
    script = bin_dir / "make"
    script.write_text(
        f"#!/bin/bash\necho \"$@\" >> '{log}'\n"
        f"cat <<'OUT'\n{output}\nOUT\nexit {exit_code}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return bin_dir, log


def make_calls(log: Path) -> int:
    """Return how often the fake make ran."""
    return len(log.read_text(encoding="utf-8").splitlines()) if log.is_file() else 0


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """Scratch git repo with the hook scripts, a .remember/ directory and src/ code."""
    root = scratch_repo(tmp_path)
    (root / ".remember").mkdir()
    (root / "src").mkdir()
    (root / "src" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "chore: code")
    return root


# ------------------------------------------------------------------ stop_test_gate


# ------------------------------------------------------------------ stop_test_gate


RED = (
    "tests/test_mod.py F\n"
    "FAILED tests/test_mod.py::test_one - assert 1 == 2\n"
    "1 failed in 0.1s"
)


def start_session(repo: Path, shape: dict[str, object] = CLAUDE) -> None:
    """Run session_start.sh, which seeds the session's tests-ok tree hash."""
    payload = {**shape, "hook_event_name": "SessionStart"}
    assert run("session_start.sh", payload, repo).returncode == 0


def edit(
    repo: Path, rel: str, text: str = "X = 2\n", shape: dict[str, object] = CLAUDE
) -> None:
    """Start the session of ``shape`` when needed, then write ``rel``."""
    key = str(shape.get("session_id", shape.get("conversation_id")))
    if not (repo / ".remember" / "tmp" / f"tests-ok.{key}").exists():
        start_session(repo, shape)
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_env(
    tmp_path: Path, *, exit_code: int, output: str
) -> tuple[dict[str, str], Path]:
    """Return an environment whose PATH starts with a fake make, and its call log."""
    bin_dir, log = fake_make(tmp_path, exit_code=exit_code, output=output)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}"}, log


def test_stop_gate_silent_on_unchanged_tree(repo: Path, tmp_path: Path) -> None:
    """Nothing changed under src/ or tests/ since session start: nothing runs."""
    start_session(repo)
    start_session(repo, CURSOR)
    (repo / "README.md").write_text("docs\n", encoding="utf-8")
    env, log = make_env(tmp_path, exit_code=1, output=RED)
    claude = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert (claude.returncode, claude.stdout) == (0, "")
    cursor = run("stop_test_gate.sh", CURSOR, repo, env)
    assert json.loads(cursor.stdout) == {}
    assert make_calls(log) == 0


def test_stop_gate_blocks_red_claude_then_rate_limits(
    repo: Path, tmp_path: Path
) -> None:
    """A changed tree + red tests block with the failing lines; a second stop within
    90 s re-sends the saved block without re-running, and a further edit makes the
    next stop re-run."""
    edit(repo, "src/mod.py")
    env, log = make_env(tmp_path, exit_code=2, output=RED)
    first = run("stop_test_gate.sh", {**CLAUDE, "hook_event_name": "Stop"}, repo, env)
    assert first.returncode == 0
    out = json.loads(first.stdout)
    assert out["decision"] == "block"
    assert "FAILED tests/test_mod.py::test_one" in out["reason"]
    assert make_calls(log) == 1
    last = repo / ".remember" / "tmp" / "stop-test-gate.s1.last"
    assert last.read_text(encoding="utf-8").split()[1] == "2"
    second = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert second.returncode == 0
    assert json.loads(second.stdout) == out
    assert make_calls(log) == 1
    edit(repo, "src/mod.py", "X = 3\n")
    third = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert json.loads(third.stdout)["decision"] == "block"
    assert make_calls(log) == 2


def test_stop_gate_green_result_skips_quietly(repo: Path, tmp_path: Path) -> None:
    """Within 90 s of a green run a stop passes quietly even after a new edit."""
    edit(repo, "src/mod.py", "X = 7\n")
    env, log = make_env(tmp_path, exit_code=0, output="1 passed")
    assert run("stop_test_gate.sh", CLAUDE, repo, env).stdout == ""
    edit(repo, "src/mod.py", "X = 8\n")
    again = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert (again.returncode, again.stdout) == (0, "")
    assert make_calls(log) == 1


def test_stop_gate_never_reuses_a_red_result(repo: Path, tmp_path: Path) -> None:
    """After the rate limit, a red session re-runs the tests; once green, the gate
    passes. Cursor gets followup_message."""
    edit(repo, "tests/test_new.py", "def test_x(): pass\n", shape=CURSOR)
    env, log = make_env(tmp_path, exit_code=2, output=RED)
    first = run("stop_test_gate.sh", CURSOR, repo, env)
    assert (
        "FAILED tests/test_mod.py::test_one"
        in json.loads(first.stdout)["followup_message"]
    )
    last = repo / ".remember" / "tmp" / "stop-test-gate.c1.last"
    last.write_text("0\n", encoding="utf-8")
    again = run("stop_test_gate.sh", CURSOR, repo, env)
    assert "followup_message" in json.loads(again.stdout)
    assert make_calls(log) == 2
    last.write_text("0\n", encoding="utf-8")
    env, log = make_env(tmp_path, exit_code=0, output="1 passed")
    green = run("stop_test_gate.sh", CURSOR, repo, env)
    assert json.loads(green.stdout) == {}
    assert make_calls(log) == 3


def test_stop_gate_ignores_sessions_without_a_start(repo: Path, tmp_path: Path) -> None:
    """A payload without an id (so no tests-ok hash) is never gated."""
    (repo / "src" / "mod.py").write_text("X = 4\n", encoding="utf-8")
    env, log = make_env(tmp_path, exit_code=2, output=RED)
    result = run("stop_test_gate.sh", {"hook_event_name": "Stop"}, repo, env)
    assert (result.returncode, result.stdout) == (0, "")
    assert make_calls(log) == 0


def test_stop_gate_catches_shell_edits(repo: Path, tmp_path: Path) -> None:
    """An edit through sed or a heredoc changes the tree hash: a src/ change since
    session start runs the tests; an unchanged tree runs nothing."""
    start_session(repo)
    assert (repo / ".remember" / "tmp" / "tests-ok.s1").is_file()
    env, log = make_env(tmp_path, exit_code=2, output=RED)
    quiet = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert (quiet.returncode, quiet.stdout) == (0, "")
    assert make_calls(log) == 0
    (repo / "src" / "mod.py").write_text("X = 5\n", encoding="utf-8")
    red = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert json.loads(red.stdout)["decision"] == "block"
    assert make_calls(log) == 1
    resent = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert json.loads(resent.stdout)["decision"] == "block"
    assert make_calls(log) == 1
    (repo / "tests").mkdir()
    (repo / "tests" / "test_new.py").write_text("def test_x(): pass\n", "utf-8")
    rerun = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert json.loads(rerun.stdout)["decision"] == "block"
    assert make_calls(log) == 2


def test_stop_gate_green_hash_ends_shell_edit_gating(
    repo: Path, tmp_path: Path
) -> None:
    """A green run records the tree hash: later stops on the same tree (and a commit
    of it) run nothing; the next shell edit runs the tests again."""
    start_session(repo, CURSOR)
    (repo / "src" / "mod.py").write_text("X = 6\n", encoding="utf-8")
    env, log = make_env(tmp_path, exit_code=0, output="1 passed")
    assert json.loads(run("stop_test_gate.sh", CURSOR, repo, env).stdout) == {}
    assert make_calls(log) == 1
    last = repo / ".remember" / "tmp" / "stop-test-gate.c1.last"
    last.write_text("0\n", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "feat: x")
    assert json.loads(run("stop_test_gate.sh", CURSOR, repo, env).stdout) == {}
    assert make_calls(log) == 1
    (repo / "src" / "mod.py").write_text("X = 7\n", encoding="utf-8")
    assert json.loads(run("stop_test_gate.sh", CURSOR, repo, env).stdout) == {}
    assert make_calls(log) == 2


def test_stop_gate_warns_on_slow_runs(repo: Path, tmp_path: Path) -> None:
    """A make tests run over 120 s logs a WARNING line on stderr."""
    edit(repo, "src/mod.py", "X = 8\n")
    env, _ = make_env(tmp_path, exit_code=0, output="1 passed")
    clock = tmp_path / "clock"
    clock.write_text("1000\n", encoding="utf-8")
    fake_date = tmp_path / "bin" / "date"
    fake_date.write_text(
        f"#!/bin/bash\nt=$(cat '{clock}')\necho \"$t\"\n"
        f"echo $((t + 125)) > '{clock}'\n",
        encoding="utf-8",
    )
    fake_date.chmod(0o755)
    result = run("stop_test_gate.sh", CLAUDE, repo, env)
    assert result.returncode == 0
    assert "WARNING: stop test gate: make tests took 125 s" in result.stderr


def test_session_start_clears_own_and_stale_gate_state(repo: Path) -> None:
    """SessionStart removes this session's run state and day-old state, and keeps the
    state of other live sessions."""
    tmp = repo / ".remember" / "tmp"
    tmp.mkdir(parents=True)
    own = [tmp / "stop-test-gate.s1.last", tmp / "agent-check.s1.out"]
    other = tmp / "stop-test-gate.live.last"
    old = tmp / "stop-test-gate.gone.last"
    for path in [*own, other, old]:
        path.write_text("x\n", encoding="utf-8")
    day_ago = time.time() - 2 * 86400
    os.utime(old, (day_ago, day_ago))
    result = run(
        "session_start.sh", {**CLAUDE, "hook_event_name": "SessionStart"}, repo
    )
    assert result.returncode == 0
    assert [p for p in [*own, old] if p.exists()] == []
    assert other.is_file()


@pytest.mark.parametrize("case", ["no_remember", "disabled", "no_make"])
def test_stop_gate_bounded_and_safe(repo: Path, tmp_path: Path, case: str) -> None:
    """A missing .remember/, the off switch and a missing make all pass."""
    edit(repo, "src/mod.py", "X = 4\n")
    env, log = make_env(tmp_path, exit_code=2, output=RED)
    payload: dict[str, object] = dict(CLAUDE)
    if case == "no_remember":
        shutil.rmtree(repo / ".remember")
    elif case == "disabled":
        env["NLP_SHAP_TEST_GATE"] = "off"
    else:
        tools = tmp_path / "tools"
        tools.mkdir()
        for name in TOOLS:
            found = shutil.which(name)
            if found:
                (tools / name).symlink_to(found)
        env = {"PATH": str(tools)}
    result = run("stop_test_gate.sh", payload, repo, env)
    assert (result.returncode, result.stdout) == (0, "")
    assert make_calls(log) == 0
    if case == "no_remember":
        assert not (repo / ".remember").exists()


def test_stop_gate_applies_to_continuations(repo: Path, tmp_path: Path) -> None:
    """A continuation (stop_hook_active) is still gated."""
    edit(repo, "src/mod.py", "X = 6\n")
    env, log = make_env(tmp_path, exit_code=2, output=RED)
    result = run("stop_test_gate.sh", {**CLAUDE, "stop_hook_active": True}, repo, env)
    assert json.loads(result.stdout)["decision"] == "block"
    assert make_calls(log) == 1


def test_run_limited_times_out() -> None:
    """The watchdog stops a command after the limit and reports 124."""
    snippet = (
        f'source "{HOOKS / "lib.sh"}" </dev/null; source "{HOOKS / "gate_lib.sh"}"; '
        'run_limited 1 sleep 20; echo "rc=$?"'
    )
    start = time.monotonic()
    result = subprocess.run(
        ["bash", "-c", snippet], capture_output=True, text=True, timeout=30, check=False
    )
    assert "rc=124" in result.stdout
    assert time.monotonic() - start < 10


# -------------------------------------------------------------------- registration


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
