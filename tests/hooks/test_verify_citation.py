"""Skill paper-notes: the citation checker reads Crossref, DataCite, arXiv or a url.

No network: ``fetch`` and ``time.sleep`` are replaced by canned responses.

@see [[.agents/skills/paper-notes/scripts/verify_citation.py]]
"""

import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.hooks._hookrun import ROOT

SCRIPT = ROOT / ".agents" / "skills" / "paper-notes" / "scripts" / "verify_citation.py"

CROSSREF = {
    "message": {
        "title": ["Attention Is All You Need"],
        "author": [{"family": "Vaswani"}, {"family": "Shazeer"}],
        "issued": {"date-parts": [[2017]]},
    }
}
ARXIV = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<title>Attention Is All
  You Need</title><published>2017-06-12T00:00:00Z</published>
<author><name>Ashish Vaswani</name></author><author><name>Noam Shazeer</name></author>
</entry></feed>"""
EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

NOTE = """---
title: "Transformer"
aliases: ["Attention Is All You Need"]
authors: ["Ashish Vaswani", "Noam Shazeer"]
year: {year}
{ids}
verified: {verified}
tags: [type/paper, theme/models]
---
# Transformer
"""


DATACITE = {
    "data": {
        "attributes": {
            "titles": [{"title": "Attention Is All You Need"}],
            "creators": [{"familyName": "Vaswani"}, {"name": "Shazeer, Noam"}],
            "publicationYear": 2017,
        }
    }
}
PAGE = b"<html><head><title>Qwen &amp; Co - Hugging Face</title></head></html>"


@pytest.fixture
def module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the checker with ``fetch`` answering from canned records."""
    spec = importlib.util.spec_from_file_location("verify_citation_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    loaded: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.calls = []
    flaky = {"n": 0}

    def fake_fetch(url: str) -> object:
        loaded.calls.append(url)
        if "id_list=0000.00000" in url:
            return loaded.Response(200, EMPTY_FEED.encode())
        if "error" in url:
            raise OSError("offline")
        if "zenodo" in url and "crossref" in url:
            return loaded.Response(404, b"")
        if "datacite" in url:
            return loaded.Response(200, json.dumps(DATACITE).encode())
        if "flaky" in url:
            flaky["n"] += 1
            return loaded.Response(
                *((429, b"", "3") if flaky["n"] < 3 else (200, PAGE))
            )
        if "down" in url:
            return loaded.Response(503, b"")
        if "gone" in url:
            return loaded.Response(404, b"")
        if "huggingface" in url:
            return loaded.Response(200, PAGE)
        if "crossref" in url:
            return loaded.Response(200, json.dumps(CROSSREF).encode())
        return loaded.Response(200, ARXIV.encode())

    monkeypatch.setattr(loaded, "fetch", fake_fetch)
    monkeypatch.setattr(loaded.time, "sleep", loaded.calls.append)
    return loaded


def _note(
    tmp_path: Path,
    ids: str,
    year: int = 2017,
    name: str = "T.md",
    verified: str = "2025-01-31",
) -> Path:
    path = tmp_path / "research" / "Models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        NOTE.format(year=year, ids=ids, verified=verified), encoding="utf-8"
    )
    return path


def test_matching_doi_and_arxiv_pass(module: ModuleType, tmp_path: Path) -> None:
    note = _note(tmp_path, 'doi: "10.5555/1"\narxiv: "1706.03762"')
    assert module.check(note) == []


def test_wrong_year_is_reported(module: ModuleType, tmp_path: Path) -> None:
    problems = module.check(_note(tmp_path, 'doi: "10.5555/1"', year=2018))
    assert len(problems) == 1
    assert "year 2017 != 2018" in problems[0]


def test_title_and_author_mismatch(module: ModuleType, tmp_path: Path) -> None:
    note = _note(tmp_path, 'arxiv: "1706.03762"')
    note.write_text(
        note
        .read_text(encoding="utf-8")
        .replace("Attention Is All You Need", "Other")
        .replace("Noam Shazeer", "Someone Else"),
        encoding="utf-8",
    )
    problems = module.check(note)
    assert any("title" in p for p in problems)
    assert any("authors" in p for p in problems)


def test_missing_record_and_ids(module: ModuleType, tmp_path: Path) -> None:
    assert module.check(_note(tmp_path, "")) == ["no doi, arxiv id or url"]
    missing = module.check(_note(tmp_path, 'arxiv: "0000.00000"'))
    assert "lookup failed" in missing[0]
    assert "lookup failed" in module.check(_note(tmp_path, 'doi: "10.1/error"'))[0]


def test_arxiv_doi_is_skipped(module: ModuleType, tmp_path: Path) -> None:
    note = _note(tmp_path, 'doi: "10.48550/arXiv.1706.03762"\narxiv: "1706.03762"')
    assert module.check(note) == []


def test_datacite_fallback_on_crossref_404(module: ModuleType, tmp_path: Path) -> None:
    note = _note(tmp_path, 'doi: "10.5281/zenodo.15413169"')
    assert module.check(note) == []
    assert any("api.datacite.org/dois/10.5281" in str(c) for c in module.calls)


def test_retry_honours_retry_after_and_is_bounded(
    module: ModuleType, tmp_path: Path
) -> None:
    url = "url: https://huggingface.co/flaky/Qwen"
    note = _note(tmp_path, url)
    note.write_text(note.read_text().replace("Transformer", "Qwen"), encoding="utf-8")
    assert module.check(note) == []
    assert [c for c in module.calls if not isinstance(c, str)] == [3.0, 3.0]
    module.calls.clear()
    problems = module.check(
        _note(tmp_path, "url: https://example.org/down", name="D.md")
    )
    assert "HTTP 503" in problems[0]
    assert (
        len([c for c in module.calls if not isinstance(c, str)]) == module.RETRIES - 1
    )


def test_url_only_checks_title_words(module: ModuleType, tmp_path: Path) -> None:
    note = _note(tmp_path, "url: https://huggingface.co/Qwen/Qwen3-8B")
    assert module.check(note) != []  # "attention is all you need" is not in the page
    note.write_text(note.read_text().replace("Attention Is All You Need", "Qwen"))
    assert module.check(note) == []
    ok = module.check(_note(tmp_path, "url: https://example.org/gone", name="G.md"))
    assert "not reachable" in ok[0]
    assert module.check(_note(tmp_path, "", name="N.md")) == ["no doi, arxiv id or url"]


def test_lenient_authors() -> None:
    spec = importlib.util.spec_from_file_location("vc_authors", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    vc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vc)
    assert vc.same_authors(
        ["Vaswani, A.", "N. Shazeer"], ["Ashish Vaswani", "Noam Shazeer"]
    )
    assert vc.same_authors(["A. Vaswani"], ["Ashish Vaswani", "Noam Shazeer", "et al"])
    assert not vc.same_authors(["Ashish Vaswani"], ["Noam Shazeer"])
    assert not vc.same_authors([], [])


def test_verified_field(module: ModuleType, tmp_path: Path) -> None:
    ids = 'arxiv: "1706.03762"'
    future = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    assert "in the future" in module.check(_note(tmp_path, ids, verified=future))[0]
    assert "not an ISO date" in module.check(_note(tmp_path, ids, verified="soon"))[0]
    bad = _note(tmp_path, ids)
    bad.write_text(
        bad.read_text().replace("verified: 2025-01-31\n", ""), encoding="utf-8"
    )
    assert module.check(bad) == ["verified: missing"]


def test_main_checks_paper_notes_only(
    module: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _note(tmp_path, 'arxiv: "1706.03762"')
    hub = tmp_path / "research" / "Research.md"
    hub.write_text("---\ntitle: Research\ntags: [type/hub]\n---\n", encoding="utf-8")
    module.ROOT = tmp_path
    assert module.paper_notes(tmp_path / "research") == [
        tmp_path / "research" / "Models" / "T.md"
    ]
    assert module.main([]) == 0
    assert capsys.readouterr().out == "OK T.md\n"
    bad = _note(tmp_path, "", name="Bad.md")
    assert module.main([str(bad)]) == 1
    weak = _note(tmp_path, "url: https://huggingface.co/x/attention", name="W.md")
    weak.write_text(weak.read_text().replace("Attention Is All You Need", "Qwen"))
    assert module.main([str(weak)]) == 0
    assert "OK (url only) W.md" in capsys.readouterr().out


def test_user_agent_names_project(
    module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.agent-kit]\nname = "demo"\n', encoding="utf-8"
    )
    monkeypatch.delenv("LITERATURE_MAILTO", raising=False)
    assert module.user_agent(tmp_path) == "demo citation check"
    monkeypatch.setenv("LITERATURE_MAILTO", "me@example.com")
    assert module.user_agent(tmp_path).endswith("(mailto:me@example.com)")
