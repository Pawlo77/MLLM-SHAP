"""Skill literature-search: harvest merges, screens and records search results.

No network: ``fetch`` and ``time.sleep`` are replaced.

@see [[.agents/skills/literature-search/scripts/harvest.py]]
"""

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tests.hooks._hookrun import ROOT

SCRIPT = ROOT / ".agents" / "skills" / "literature-search" / "scripts" / "harvest.py"

PROTOCOL = """name: demo
since: "2020-01-01"
openalex_queries: [shapley text]
arxiv_queries: ['abs:shapley']
seeds: ["2407.10114"]
include:
  - ['shapley']
  - ['text|language']
exclude: [federated]
"""

OPENALEX_WORK = {
    "title": "Shapley Values for Text",
    "publication_year": 2024,
    "doi": "https://doi.org/10.1/ABC",
    "ids": {"openalex": "W1", "doi": "https://doi.org/10.48550/arXiv.2401.00001"},
    "primary_location": {"source": {"display_name": "ACL"}},
    "abstract_inverted_index": {"language": [1], "Explaining": [0]},
}
ARXIV_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2401.00001v2</id><title>Shapley Values
 for Text</title><published>2024-01-01T00:00:00Z</published>
<summary>Explaining language models.</summary></entry></feed>"""
S2 = {
    "data": [
        {
            "citingPaper": {
                "title": "Federated Shapley text",
                "year": 2025,
                "externalIds": {"DOI": "10.2/x"},
            }
        }
    ]
}


@pytest.fixture
def harvest(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load harvest.py with canned API answers and no sleeping."""
    spec = importlib.util.spec_from_file_location("harvest_under_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_fetch(url: str, tries: int = 5) -> bytes:
        if "api.openalex.org/works/doi:" in url:
            return b'{"id": "https://openalex.org/W9"}'
        if "openalex" in url:
            return json.dumps({"results": [OPENALEX_WORK]}).encode()
        if "arxiv.org" in url:
            return ARXIV_FEED.encode()
        return json.dumps(S2).encode()

    monkeypatch.setattr(module, "fetch", fake_fetch)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def test_record_keys_fold_arxiv_doi(harvest: ModuleType) -> None:
    keys = harvest.record_keys({
        "doi": "https://doi.org/10.48550/arXiv.2401.00001",
        "title": "A Title!",
    })
    assert keys == {"arxiv:2401.00001", "title:a title"}


def test_merge_joins_on_any_key(harvest: ModuleType) -> None:
    merged = harvest.merge([
        {"title": "Paper", "arxiv": "2401.00001v1", "src": ["a"]},
        {"title": "Other name", "arxiv": "2401.00001", "venue": "ACL", "src": ["b"]},
        {"title": "paper", "doi": "10.1/x", "src": ["c"]},
    ])
    assert len(merged) == 1
    assert merged[0]["venue"] == "ACL"
    assert merged[0]["src"] == ["a", "b", "c"]


def test_screen_counts_each_outcome(harvest: ModuleType) -> None:
    records = [
        {"title": "Shapley for language"},
        {"title": "Shapley federated language"},
        {"title": "Shapley for images"},
        {"title": "Known paper"},
    ]
    kept, counts = harvest.screen(
        records, [["shapley"], ["language"]], ["federated"], {"title:known paper"}
    )
    assert [r["title"] for r in kept] == ["Shapley for language"]
    assert counts == {
        "records": 4,
        "covered": 1,
        "excluded": 1,
        "off_topic": 1,
        "kept": 1,
    }


def test_covered_keys_read_note_frontmatter(
    harvest: ModuleType, tmp_path: Path
) -> None:
    note = tmp_path / "research" / "Area" / "P.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        '---\ntitle: P\naliases: ["Long Title"]\narxiv: "2401.00001"\n---\n',
        encoding="utf-8",
    )
    (tmp_path / "research" / "plain.md").write_text("no frontmatter", encoding="utf-8")
    keys = harvest.covered_keys(tmp_path / "research")
    assert {"title:p", "title:long title", "arxiv:2401.00001"} <= keys
    assert harvest.covered_keys(tmp_path / "missing") == set()


def test_openalex_record_reassembles_abstract(harvest: ModuleType) -> None:
    rec = harvest.openalex_record(OPENALEX_WORK, "openalex:q")
    assert rec["abstract"] == "Explaining language"
    assert rec["arxiv"] == "2401.00001"
    assert rec["venue"] == "ACL"


def test_main_fetches_and_screens(harvest: ModuleType, tmp_path: Path) -> None:
    protocol = tmp_path / "demo.yaml"
    protocol.write_text(PROTOCOL, encoding="utf-8")
    out = tmp_path / "out"
    research = tmp_path / "research"
    assert (
        harvest.main([str(protocol), "--out", str(out), "--research", str(research)])
        == 0
    )
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    assert counts["retrieved"] == 4  # keyword, seed citation, arXiv, Semantic Scholar
    assert counts["excluded"] == 1
    assert counts["kept"] == 1
    with (out / "candidates.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["title"] == "Shapley Values for Text"
    assert rows[0]["tier"] == ""
    assert harvest.main([str(protocol), "--out", str(out), "--stage", "screen"]) == 0


def test_user_agent_names_project(
    harvest: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.agent-kit]\nname = "demo"\n', encoding="utf-8"
    )
    monkeypatch.delenv("LITERATURE_MAILTO", raising=False)
    assert harvest.user_agent(tmp_path) == "demo literature search"
