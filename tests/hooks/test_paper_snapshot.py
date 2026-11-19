"""Skill paper-notes: full-text snapshots mirror research/ under fulltext/.

No network: ``fetch`` is replaced by canned responses.

@see [[.agents/skills/paper-notes/scripts/snapshot.py]]
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

SCRIPT = ROOT / ".agents" / "skills" / "paper-notes" / "scripts" / "snapshot.py"

NOTE = """---
title: "Transformer"
arxiv: "1706.03762"
url: "https://example.org/transformer"
tags: [type/paper, theme/models]
---
# Transformer
"""

PAGE = "<html><body><script>x()</script>" + "<p>word</p>" * 2000 + "</body></html>"


@pytest.fixture
def snap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the script with research/ and fulltext/ under a temporary root."""
    spec = importlib.util.spec_from_file_location("paper_snapshot", SCRIPT)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "RESEARCH", tmp_path / "research")
    monkeypatch.setattr(module, "FULLTEXT", tmp_path / "fulltext")
    return module


def write_note(root: Path, rel: str) -> Path:
    """Create a paper note at ``research/<rel>``."""
    path = root / "research" / rel
    path.parent.mkdir(parents=True)
    path.write_text(NOTE, encoding="utf-8")
    return path


def test_snapshot_mirrors_note_path_and_falls_back_to_url(
    snap: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed arXiv download falls back to the HTML url at the mirror path."""
    note = write_note(tmp_path, "Models/Attention/Transformer.md")

    def fake(url: str) -> tuple[bytes, str]:
        if "arxiv" in url:
            raise OSError("down")
        return PAGE.encode(), "text/html"

    monkeypatch.setattr(snap, "fetch", fake)
    assert snap.snapshot(note, None, False).startswith("wrote")
    out = tmp_path / "fulltext" / "Models" / "Attention" / "Transformer.txt"
    text = out.read_text(encoding="utf-8")
    assert "# source: https://example.org/transformer" in text
    assert "x()" not in text
    assert snap.snapshot(note, None, False).startswith("exists")


def test_short_text_is_rejected(
    snap: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abstract-only page is not a full text."""
    note = write_note(tmp_path, "Models/Transformer.md")
    monkeypatch.setattr(snap, "fetch", lambda url: (b"<p>abstract</p>", "text/html"))
    assert snap.snapshot(note, None, False).startswith("FAIL")


def test_sync_moves_snapshot_after_note_moves(snap: ModuleType, tmp_path: Path) -> None:
    """A note moved to another theme takes its snapshot along."""
    write_note(tmp_path, "New theme/Transformer.md")
    old = tmp_path / "fulltext" / "Old theme" / "Transformer.txt"
    old.parent.mkdir(parents=True)
    old.write_text("text", encoding="utf-8")
    snap.sync()
    assert not old.exists()
    assert (tmp_path / "fulltext" / "New theme" / "Transformer.txt").exists()


def test_supplement_field_is_appended(
    snap: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild keeps the supplement listed in the note."""
    note = write_note(tmp_path, "Models/Transformer.md")
    url = "https://example.org/supp"
    text = note.read_text(encoding="utf-8").replace(
        "tags:", f"supplement: {url}\ntags:"
    )
    note.write_text(text, encoding="utf-8")
    monkeypatch.setattr(snap, "fetch", lambda _: (PAGE.encode(), "text/html"))
    snap.snapshot(note, None, False)
    out = tmp_path / "fulltext" / "Models" / "Transformer.txt"
    assert f"# supplement: {url}" in out.read_text(encoding="utf-8")


def test_all_writes_every_missing_snapshot(
    snap: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--all`` rebuilds a fresh clone, which has no fulltext/."""
    write_note(tmp_path, "A/One.md")
    write_note(tmp_path, "B/Two.md")
    monkeypatch.setattr(snap, "fetch", lambda _: (PAGE.encode(), "text/html"))
    assert snap.main(["--all"]) == 0
    assert (tmp_path / "fulltext" / "A" / "One.txt").exists()
    assert (tmp_path / "fulltext" / "B" / "Two.txt").exists()


def test_pdf_flag_reads_a_local_file(snap: ModuleType, tmp_path: Path) -> None:
    """A downloaded paper can be passed as a path; nothing goes to the network."""
    note = write_note(tmp_path, "Area/Transformer.md")
    local = tmp_path / "paper.html"
    local.write_text(PAGE, encoding="utf-8")
    assert snap.main([str(note), "--pdf", str(local)]) == 0
    assert "word\nword" in snap.target(note).read_text(encoding="utf-8")


def test_plain_text_is_kept_verbatim(snap: ModuleType) -> None:
    """Code supplements keep their angle brackets."""
    assert snap.decode(b"a < b and c > d", "text/plain; charset=utf-8") == (
        "a < b and c > d"
    )
