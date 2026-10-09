"""Output shapes of the shared agent hooks for Cursor- and Claude-shaped payloads.

@see [[scripts/hooks/lib.sh]]
@see [[scripts/hooks/format_after_edit.sh]]
@see [[scripts/hooks/statusline.sh]]
"""

import json
import subprocess
from pathlib import Path

import pytest

from tests.hooks._hookrun import (
    CLAUDE,
    CURSOR,
    HAVE_TOOLS,
    HOOKS,
    run_hook,
    scratch_repo,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="needs bash and jq")


def _lib(payload: dict[str, object], snippet: str) -> subprocess.CompletedProcess[str]:
    """Source lib.sh with ``payload`` on stdin, then run ``snippet``."""
    return subprocess.run(
        ["bash", "-c", f'source "{HOOKS / "lib.sh"}"; {snippet}'],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({**CURSOR, "file_path": "/a/b.py"}, "/a/b.py"),
        ({**CLAUDE, "tool_input": {"file_path": "/c/d.py"}}, "/c/d.py"),
        ({**CLAUDE, "tool_input": {"notebook_path": "/e.ipynb"}}, "/e.ipynb"),
    ],
)
def test_edited_file_both_shapes(payload: dict[str, object], expected: str) -> None:
    """edited_file reads Cursor's top-level and Claude's tool_input path."""
    assert _lib(payload, "edited_file").stdout.strip() == expected


@pytest.mark.parametrize(
    "payload",
    [
        {**CURSOR, "command": "git status"},
        {**CLAUDE, "tool_input": {"command": "git status"}},
    ],
)
def test_shell_command_both_shapes(payload: dict[str, object]) -> None:
    """shell_command reads Cursor's command and Claude's tool_input.command."""
    assert _lib(payload, "shell_command").stdout.strip() == "git status"


def test_emit_context_shapes() -> None:
    """Cursor gets additional_context; Claude gets text or hookSpecificOutput."""
    cur = json.loads(_lib(CURSOR, "emit_context PostToolUse hello").stdout)
    assert cur == {"additional_context": "hello"}
    start = _lib(CLAUDE, "emit_context SessionStart hello")
    assert start.stdout == "hello\n"
    post = json.loads(_lib(CLAUDE, "emit_context PostToolUse hello short").stdout)
    assert post == {
        "systemMessage": "short",
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": "hello",
        },
    }


def test_emit_blocking_shapes() -> None:
    """Claude blocks with stderr + exit 2; Cursor gets context and exit 0."""
    cur = _lib(CURSOR, "emit_blocking 'broken link'")
    assert cur.returncode == 0
    ctx = json.loads(cur.stdout)["additional_context"]
    assert ctx.startswith("Fix these before continuing:")
    assert "broken link" in ctx
    cla = _lib(CLAUDE, "emit_blocking 'broken link'")
    assert cla.returncode == 2
    assert cla.stderr.strip() == "broken link"
    assert cla.stdout == ""


def test_emit_deny_shapes() -> None:
    """Cursor deny is flat; Claude deny is a PreToolUse permissionDecision."""
    cur = json.loads(_lib(CURSOR, "emit_deny nope").stdout)
    assert cur == {
        "permission": "deny",
        "user_message": "nope",
        "agent_message": "nope",
    }
    cla = json.loads(_lib(CLAUDE, "emit_deny nope").stdout)
    assert cla["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert cla["hookSpecificOutput"]["permissionDecisionReason"] == "nope"


def test_format_after_edit_formats_python(tmp_path: Path) -> None:
    """An edited .py file is Ruff-formatted for both payload shapes."""
    for payload_of in (
        lambda p: {**CURSOR, "file_path": str(p)},
        lambda p: {**CLAUDE, "tool_input": {"file_path": str(p)}},
    ):
        target = tmp_path / "probe_mod.py"
        target.write_text("x=1\n", encoding="utf-8")
        proc = run_hook(HOOKS / "format_after_edit.sh", payload_of(target))
        assert proc.returncode == 0
        assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_format_after_edit_keeps_unused_imports_and_reports(tmp_path: Path) -> None:
    """An import added before the edit that uses it survives (F401 is never fixed);
    the hook names the fixed rule codes and the reformat in one line, and is silent
    when nothing changed."""
    target = tmp_path / "probe_imports.py"
    target.write_text("import sys\nimport os\nx=1\n", encoding="utf-8")
    payload = {**CLAUDE, "tool_input": {"file_path": str(target)}}
    proc = run_hook(HOOKS / "format_after_edit.sh", payload)
    assert proc.returncode == 0
    assert target.read_text(encoding="utf-8") == "import os\nimport sys\n\nx = 1\n"
    line = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    assert line.startswith("format_after_edit: ")
    assert line.endswith("probe_imports.py: ruff fixed I001; reformatted")
    again = run_hook(HOOKS / "format_after_edit.sh", payload)
    assert (again.returncode, again.stdout) == (0, "")


def test_format_after_edit_strips_watermarks(tmp_path: Path) -> None:
    """AI attribution lines are removed from any edited file, before formatting."""
    target = tmp_path / "note.md"
    target.write_text(
        "# Note\n\nBody.\n\nCo-Authored-By: Claude <noreply@anthropic.com>\n",
        encoding="utf-8",
    )
    proc = run_hook(
        HOOKS / "format_after_edit.sh",
        {**CLAUDE, "tool_input": {"file_path": str(target)}},
    )
    assert proc.returncode == 0
    assert target.read_text(encoding="utf-8") == "# Note\n\nBody.\n"
    line = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
    assert line.endswith("note.md: removed 1 AI attribution line(s)")


def test_statusline(tmp_path: Path) -> None:
    """Status line prints project, branch and dirty-file count."""
    repo = scratch_repo(tmp_path)
    (repo / "new.txt").write_text("x\n", encoding="utf-8")
    proc = run_hook(repo / "scripts" / "hooks" / "statusline.sh", CLAUDE, cwd=repo)
    assert proc.stdout.strip() == "nlp-shap main 1"


def _codex_patch(*lines: str) -> dict[str, object]:
    """A Codex PostToolUse payload for apply_patch with the given patch lines."""
    patch = "\n".join(["*** Begin Patch", *lines, "*** End Patch"])
    return {**CLAUDE, "tool_name": "apply_patch", "tool_input": {"command": patch}}


def test_edited_file_codex_patch() -> None:
    """edited_file lists the Update and Add paths of a Codex patch, one per line."""
    payload = _codex_patch(
        "*** Update File: src/a.py",
        "@@\n-a\n+b",
        "*** Add File: tests/b.py",
        "+x",
        "*** Delete File: c.py",
    )
    assert _lib(payload, "edited_file").stdout == "src/a.py\ntests/b.py\n"


def test_format_after_edit_codex_patch(tmp_path: Path) -> None:
    """Every .py file of a Codex patch is Ruff-formatted."""
    one, two = tmp_path / "one_mod.py", tmp_path / "two_mod.py"
    for target in (one, two):
        target.write_text("x=1\n", encoding="utf-8")
    payload = _codex_patch(f"*** Update File: {one}", f"*** Add File: {two}")
    assert run_hook(HOOKS / "format_after_edit.sh", payload).returncode == 0
    assert one.read_text(encoding="utf-8") == "x = 1\n"
    assert two.read_text(encoding="utf-8") == "x = 1\n"
