"""Session-start hook: project state plus committed memory for Cursor.

@see [[scripts/hooks/session_start.sh]]
"""

import json
from pathlib import Path

import pytest

from tests.hooks._hookrun import CLAUDE, CURSOR, HAVE_TOOLS, run_hook, scratch_repo

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="needs bash and jq")

HANDOFF = "Handoff marker: next run Phase 6."
IDENTITY = "Identity marker: engineering partner on demo-kit."


def _repo_with_memory(tmp_path: Path) -> Path:
    repo = scratch_repo(tmp_path)
    mem = repo / ".remember"
    mem.mkdir()
    (mem / "remember.md").write_text(f"# Handoff\n\n{HANDOFF}\n", encoding="utf-8")
    (mem / "identity.md").write_text(f"# Identity\n\n{IDENTITY}\n", encoding="utf-8")
    lines = [f"recent line {i}" for i in range(100)]
    (mem / "recent.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (mem / "today-2026-10-06.md").write_text("old day marker\n", encoding="utf-8")
    (mem / "today-2026-10-07.md").write_text("new day marker\n", encoding="utf-8")
    return repo


def _start(repo: Path, payload: dict[str, object]) -> str:
    proc = run_hook(repo / "scripts" / "hooks" / "session_start.sh", payload, cwd=repo)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_cursor_gets_state_and_memory(tmp_path: Path) -> None:
    """Cursor context carries version, handoff, identity and memory tails."""
    repo = _repo_with_memory(tmp_path)
    ctx = json.loads(_start(repo, CURSOR))["additional_context"]
    assert "## demo-kit session state" in ctx
    assert "make check" in ctx
    assert HANDOFF in ctx
    assert IDENTITY in ctx
    assert ctx.index(HANDOFF) < ctx.index(IDENTITY)
    assert "recent line 99" in ctx
    assert "recent line 39" not in ctx  # only the last 60 lines
    assert "new day marker" in ctx
    assert "old day marker" not in ctx


def test_handoff_of_real_size_is_inlined(tmp_path: Path) -> None:
    """A handoff of a few KB (remember.md is about 2.2 KB) reaches Cursor whole."""
    repo = _repo_with_memory(tmp_path)
    handoff = HANDOFF + "\n" + "h" * 3000
    (repo / ".remember" / "remember.md").write_text(handoff, encoding="utf-8")
    ctx = json.loads(_start(repo, CURSOR))["additional_context"]
    assert handoff in ctx


def test_oversized_memory_file_is_referenced_not_inlined(tmp_path: Path) -> None:
    """A handoff over 4 KB is replaced by its path and size."""
    repo = _repo_with_memory(tmp_path)
    big = "B" * 5000
    (repo / ".remember" / "remember.md").write_text(big, encoding="utf-8")
    ctx = json.loads(_start(repo, CURSOR))["additional_context"]
    assert big not in ctx
    assert ".remember/remember.md is 5000 bytes" in ctx
    assert IDENTITY in ctx


def test_total_memory_cap(tmp_path: Path) -> None:
    """Inlined memory stays under the 12 KB total cap."""
    repo = _repo_with_memory(tmp_path)
    mem = repo / ".remember"
    (mem / "identity.md").write_text("I" * 4000, encoding="utf-8")
    (mem / "recent.md").write_text(("R" * 99 + "\n") * 60, encoding="utf-8")
    (mem / "today-2026-10-07.md").write_text(("T" * 99 + "\n") * 40, encoding="utf-8")
    ctx = json.loads(_start(repo, CURSOR))["additional_context"]
    memory = ctx.split("## Memory and context", 1)[1]
    assert len(memory.encode()) < 12288 + 1024  # payload cap plus headings
    assert "over the inline cap" in memory


def test_claude_gets_plain_text_without_memory(tmp_path: Path) -> None:
    """Claude Code reads SessionStart stdout; memory comes from the plugin, so there
    is neither inlined memory nor a pointer line, and no hooks summary."""
    repo = _repo_with_memory(tmp_path)
    out = _start(repo, CLAUDE)
    assert out.startswith("## demo-kit session state")
    assert "exit criterion: make check" in out
    assert "session key" in out
    assert HANDOFF not in out
    assert "- memory:" not in out
    assert "- hooks:" not in out


def test_dirty_files_capped_at_five(tmp_path: Path) -> None:
    """The banner gives the dirty-file count and only the first five files."""
    repo = _repo_with_memory(tmp_path)
    for i in range(8):
        (repo / f"dirty_{i}.txt").write_text("x\n", encoding="utf-8")
    out = _start(repo, CLAUDE)
    assert "uncommitted changes (9 files; first 5):" in out
    assert sum(1 for line in out.splitlines() if line.startswith("    ?? ")) == 5


def test_project_extension_adds_state_and_memory(tmp_path: Path) -> None:
    """The project-owned ``project_state.sh`` adds banner lines and memory items."""
    repo = _repo_with_memory(tmp_path)
    (repo / "scripts" / "hooks" / "project_state.sh").write_text(
        'project_state() { echo "- extra: state line"; }\n'
        'project_memory() { mem_item "Extra" "x.md" 100 "extra memory"; }\n',
        encoding="utf-8",
    )
    assert "- extra: state line" in _start(repo, CLAUDE)
    ctx = json.loads(_start(repo, CURSOR))["additional_context"]
    assert "extra memory" in ctx
