"""Harvest and screen literature records for a domain search protocol (YAML).

Usage: ``uv run python .agents/skills/literature-search/scripts/harvest.py PROTOCOL.yaml
--out DIR [--stage fetch|screen|all] [--research DIR]``.
Fetch queries OpenAlex (keywords and forward citations of seeds), arXiv and Semantic
Scholar citations, retrying with backoff, and writes ``DIR/raw.json``. Screen merges
duplicates by DOI, arXiv id and title, drops papers already noted under ``research/``,
applies the protocol's include groups and excludes, and writes ``DIR/candidates.csv``
for manual title and abstract screening plus ``DIR/counts.json``. Fetch needs network;
``LITERATURE_MAILTO`` joins the OpenAlex polite pool.
"""

import argparse
import csv
import json
import os
import re
import time
import tomllib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET  # nosec B405 - parses the arXiv API response only
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
"""Repository root, four levels above this script."""

ATOM = {"a": "http://www.w3.org/2005/Atom"}
OPENALEX_FIELDS = (
    "id,title,publication_year,doi,ids,primary_location,abstract_inverted_index"
)
SHEET = [
    "tier",
    "decision",
    "reason",
    "title",
    "year",
    "venue",
    "doi",
    "arxiv",
    "src",
    "abstract",
]
Record = dict[str, Any]


def norm_title(text: str) -> str:
    """Lower-case a title and collapse everything but letters and digits to spaces."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _doi(value: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (value or "").strip()).lower()


def record_keys(rec: Record) -> set[str]:
    """Identity keys of a record: DOI, arXiv id (also from an arXiv DOI) and title."""
    keys = set()
    doi = _doi(str(rec.get("doi") or ""))
    arxiv = str(rec.get("arxiv") or "")
    if doi.startswith("10.48550/arxiv."):
        arxiv = arxiv or doi.removeprefix("10.48550/arxiv.")
    elif doi:
        keys.add(f"doi:{doi}")
    if arxiv:
        keys.add("arxiv:" + re.sub(r"v\d+$", "", arxiv.lower()))
    if title := norm_title(str(rec.get("title") or "")):
        keys.add(f"title:{title}")
    return keys


def merge(records: list[Record]) -> list[Record]:
    """Join records sharing any identity key; the first keeps the others' sources."""
    merged: list[Record] = []
    index: dict[str, Record] = {}
    for rec in records:
        keys = record_keys(rec)
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            hit = {**rec, "src": list(rec.get("src", []))}
            merged.append(hit)
        else:
            for field in ("doi", "arxiv", "abstract", "venue", "year"):
                hit[field] = hit.get(field) or rec.get(field)
            hit["src"] += [s for s in rec.get("src", []) if s not in hit["src"]]
        index.update(dict.fromkeys(keys | record_keys(hit), hit))
    return merged


def covered_keys(research: Path) -> set[str]:
    """Identity keys of every note under research/ (doi, arxiv, title and aliases)."""
    keys: set[str] = set()
    for path in sorted(research.rglob("*.md")) if research.is_dir() else []:
        parts = path.read_text(encoding="utf-8").split("---", 2)
        meta = yaml.safe_load(parts[1]) if len(parts) == 3 else None
        if not isinstance(meta, dict):
            continue
        titles = [meta.get("title", ""), *(meta.get("aliases") or [])]
        for title in titles:
            keys |= record_keys({"title": title})
        keys |= record_keys({"doi": meta.get("doi"), "arxiv": meta.get("arxiv")})
    return keys


def screen(
    records: list[Record],
    include: list[list[str]],
    exclude: list[str],
    covered: set[str],
) -> tuple[list[Record], dict[str, int]]:
    """Keep records matching a pattern of every include group and no exclude."""
    counts = {
        "records": len(records),
        "covered": 0,
        "excluded": 0,
        "off_topic": 0,
        "kept": 0,
    }
    kept = []
    for rec in records:
        text = f"{rec.get('title') or ''} {rec.get('abstract') or ''}"
        if record_keys(rec) & covered:
            counts["covered"] += 1
        elif any(re.search(p, text, re.IGNORECASE) for p in exclude):
            counts["excluded"] += 1
        elif all(
            any(re.search(p, text, re.IGNORECASE) for p in group) for group in include
        ):
            kept.append(rec)
        else:
            counts["off_topic"] += 1
    counts["kept"] = len(kept)
    return kept, counts


def write_sheet(records: list[Record], path: Path) -> None:
    """Write the screening sheet; tier, decision and reason are filled by hand."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, SHEET, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({**rec, "src": "; ".join(rec.get("src", []))})


def user_agent(root: Path = ROOT) -> str:
    """Return the HTTP user agent: project name, plus ``LITERATURE_MAILTO`` if set."""
    name = root.name
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        name = data.get("tool", {}).get("agent-kit", {}).get("name", name)
    mailto = os.environ.get("LITERATURE_MAILTO", "")
    return f"{name} literature search" + (f" (mailto:{mailto})" if mailto else "")


def fetch(url: str, tries: int = 5) -> bytes:
    """Return the body of ``url``; back off and retry, empty bytes after ``tries``."""
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
                return bytes(response.read())
        except OSError:
            time.sleep(10 * (attempt + 1))  # 429 and transient errors: back off
    return b""


def openalex_record(rec: Record, src: str) -> Record:
    """Map an OpenAlex work to a record; the arXiv id may hide in any of its ids."""
    inverted = rec.get("abstract_inverted_index") or {}
    words = sorted((p, w) for w, ps in inverted.items() for p in ps)
    source = (rec.get("primary_location") or {}).get("source") or {}
    found = re.search(
        r"arxiv[./:a-z]*?(\d{4}\.\d{4,5})",
        " ".join(map(str, (rec.get("ids") or {}).values())),
        re.IGNORECASE,
    )
    arxiv = found.group(1) if found else ""
    return {
        "title": rec.get("title"),
        "year": rec.get("publication_year"),
        "doi": rec.get("doi"),
        "arxiv": arxiv,
        "venue": source.get("display_name"),
        "abstract": " ".join(w for _, w in words)[:1500],
        "src": [src],
    }


def fetch_openalex(protocol: Record) -> list[Record]:
    """Keyword queries since the protocol date, then works citing each seed."""
    base = "https://api.openalex.org/works?"
    since = f"from_publication_date:{protocol['since']}"
    records = []
    for query in protocol.get("openalex_queries", []):
        params = {
            "search": query,
            "filter": since,
            "per-page": 50,
            "select": OPENALEX_FIELDS,
        }
        data = json.loads(fetch(base + urllib.parse.urlencode(params)) or b"{}")
        records += [
            openalex_record(r, f"openalex:{query}") for r in data.get("results", [])
        ]
    for seed in protocol.get("seeds", []):
        doi = seed if seed.startswith("10.") else f"10.48550/arXiv.{seed}"
        work = json.loads(
            fetch(f"https://api.openalex.org/works/doi:{doi}?select=id") or b"{}"
        )
        if wid := str(work.get("id", "")).rsplit("/", 1)[-1]:
            params = {
                "filter": f"cites:{wid}",
                "per-page": 200,
                "select": OPENALEX_FIELDS,
            }
            data = json.loads(fetch(base + urllib.parse.urlencode(params)) or b"{}")
            records += [
                openalex_record(r, f"cites:{seed}") for r in data.get("results", [])
            ]
    return records


def fetch_arxiv(protocol: Record) -> list[Record]:
    """Newest 150 arXiv records per query, three seconds apart."""
    records = []
    for query in protocol.get("arxiv_queries", []):
        params = {
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 150,
        }
        body = fetch(
            "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
        )
        feed = ET.fromstring(body or b"<feed/>")  # nosec B314 - arXiv API response
        records.extend(
            {
                "title": " ".join(entry.findtext("a:title", "", ATOM).split()),
                "year": entry.findtext("a:published", "", ATOM)[:4],
                "arxiv": entry.findtext("a:id", "", ATOM).rsplit("/", 1)[-1],
                "abstract": " ".join(entry.findtext("a:summary", "", ATOM).split()),
                "src": [f"arxiv:{query}"],
            }
            for entry in feed.findall("a:entry", ATOM)
        )
        time.sleep(3)  # arXiv asks for one request every three seconds
    return records


def fetch_s2_citations(protocol: Record) -> list[Record]:
    """Semantic Scholar papers citing each seed (rate-limited; retried with backoff)."""
    records = []
    fields = "title,year,venue,externalIds,abstract"
    for seed in protocol.get("seeds", []):
        ident = f"DOI:{seed}" if seed.startswith("10.") else f"arXiv:{seed}"
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"{ident}/citations?fields={fields}&limit=1000"
        )
        for item in json.loads(fetch(url) or b"{}").get("data", []):
            paper = item.get("citingPaper") or {}
            ids = paper.get("externalIds") or {}
            records.append({
                **paper,
                "doi": ids.get("DOI"),
                "arxiv": ids.get("ArXiv"),
                "src": [f"s2:{seed}"],
            })
        time.sleep(3)
    return records


def main(argv: list[str] | None = None) -> int:
    """Run the requested stages and print the screening counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=["fetch", "screen", "all"], default="all")
    parser.add_argument("--research", type=Path, default=ROOT / "research")
    args = parser.parse_args(argv)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw.json"
    if args.stage in ("fetch", "all"):
        records = (
            fetch_openalex(protocol)
            + fetch_arxiv(protocol)
            + fetch_s2_citations(protocol)
        )
        raw.write_text(json.dumps(records), encoding="utf-8")
    if args.stage in ("screen", "all"):
        records = merge(json.loads(raw.read_text(encoding="utf-8")))
        kept, counts = screen(
            records,
            protocol["include"],
            protocol.get("exclude", []),
            covered_keys(args.research),
        )
        counts["retrieved"] = len(json.loads(raw.read_text(encoding="utf-8")))
        write_sheet(kept, args.out / "candidates.csv")
        (args.out / "counts.json").write_text(
            json.dumps(counts, indent=2), encoding="utf-8"
        )
        print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
