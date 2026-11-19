"""Check that each paper note cites a source that exists, with matching metadata.

Usage: ``uv run python <this script> [NOTE ...]`` (default: every note under
``research/`` tagged ``type/paper``). Needs network. Per note it requires ``verified``
(ISO date, not in the future), then looks up ``doi`` (Crossref, else DataCite on 404)
and ``arxiv`` (arXiv API) and compares the official title (first alias, else
``title``), the author surnames (initials and ``et al`` tolerated) and, for a DOI, the
year. A note with neither id is checked through ``url``: it must answer 200 after
redirects and the title words must occur in the page title or Hugging Face model id.
Prints ``OK``, ``OK (url only)`` (weaker: the source exists, metadata is unchecked) or
``FAIL``; exits 1 on any ``FAIL``. HTTP 429 and 5xx are retried with bounded backoff.
"""

import datetime as dt
import html
import json
import os
import re
import sys
import time
import tomllib
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET  # nosec B405 - parses the arXiv API response only
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[4]
"""Repository root, four levels above this script."""

ATOM = {"a": "http://www.w3.org/2005/Atom"}
PAPER_TAG = "type/paper"
"""Tag that marks a paper note; hubs and folder notes carry other ``type/`` tags."""

Lookup = Callable[[str], tuple[str, list[str], int]]


def user_agent(root: Path = ROOT) -> str:
    """Return the HTTP user agent: project name, plus ``LITERATURE_MAILTO`` if set."""
    name = root.name
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        name = data.get("tool", {}).get("agent-kit", {}).get("name", name)
    mailto = os.environ.get("LITERATURE_MAILTO", "")
    return f"{name} citation check" + (f" (mailto:{mailto})" if mailto else "")


class Response(NamedTuple):
    """One HTTP answer: status, body and the ``Retry-After`` header (may be empty)."""

    status: int
    body: bytes
    retry_after: str = ""


class NotFoundError(LookupError):
    """The server answered 404."""


RETRIES = 4
"""Attempts per URL for HTTP 429 and 5xx."""
MAX_WAIT = 30.0
"""Longest pause in seconds between attempts, whatever ``Retry-After`` says."""


def fetch(url: str) -> Response:
    """Return the answer of ``url`` after redirects (one attempt): the network call."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            return Response(response.status, response.read(1 << 19))
    except urllib.error.HTTPError as exc:
        return Response(exc.code, b"", exc.headers.get("Retry-After", ""))


def get(url: str) -> bytes:
    """Return the body of ``url``; retry 429/5xx with backoff, raise on other errors."""
    for attempt in range(RETRIES):
        answer = fetch(url)
        if answer.status == 200:
            return answer.body
        if answer.status == 404:
            raise NotFoundError(url)
        if answer.status != 429 and answer.status < 500:
            break
        if attempt < RETRIES - 1:
            after = answer.retry_after
            wait = float(after) if after.isdigit() else 2**attempt
            time.sleep(min(wait, MAX_WAIT))
    msg = f"HTTP {answer.status} from {url}"
    raise OSError(msg)


def norm(text: str) -> str:
    """Fold accents and case; keep letters and digits separated by single spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def surname(name: str) -> str:
    """Return the surname of ``Given Family``, ``Family, Given`` or ``A. Family``."""
    name = name.split(",")[0] if "," in name else name
    words = [w for w in norm(name).split() if len(w) > 1] or norm(name).split()
    return words[-1] if words else ""


def crossref(doi: str) -> tuple[str, list[str], int]:
    """Return title, authors and year of a DOI; DataCite if Crossref has no record."""
    try:
        body = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    except NotFoundError:
        return datacite(doi)
    message = json.loads(body)["message"]
    authors = [a.get("family", a.get("name", "")) for a in message.get("author", [])]
    year = message["issued"]["date-parts"][0][0]
    return message["title"][0], authors, int(year)


def datacite(doi: str) -> tuple[str, list[str], int]:
    """Return title, author names and year of a DOI from DataCite."""
    attrs = json.loads(get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}"))
    attrs = attrs["data"]["attributes"]
    names = [
        c.get("familyName") or c.get("name", "") for c in attrs.get("creators", [])
    ]
    return attrs["titles"][0]["title"], names, int(attrs["publicationYear"])


def arxiv(arxiv_id: str) -> tuple[str, list[str], int]:
    """Return title, author names and first-version year of an arXiv id."""
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    root = ET.fromstring(get(url))  # nosec B314 - arXiv API response
    entry = root.find("a:entry", ATOM)
    title = entry.findtext("a:title", "", ATOM) if entry is not None else ""
    if entry is None or not title.strip():
        msg = f"arXiv has no record {arxiv_id}"
        raise LookupError(msg)
    names = [a.findtext("a:name", "", ATOM) for a in entry.findall("a:author", ATOM)]
    year = int(entry.findtext("a:published", "0000", ATOM)[:4])
    return " ".join(title.split()), names, year


SOURCES: dict[str, Lookup] = {"doi": crossref, "arxiv": arxiv}
"""Frontmatter key and the registry that resolves it."""


def frontmatter(path: Path) -> dict[str, Any]:
    """Return a note's YAML frontmatter, or an empty mapping when it has none."""
    parts = path.read_text(encoding="utf-8").split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        return {}
    meta = yaml.safe_load(parts[1])
    return meta if isinstance(meta, dict) else {}


def same_authors(found: list[str], mine: list[str]) -> bool:
    """Tell whether two author lists agree: same first surname, the shorter a subset.

    Lists differ in initials, name order, ``et al`` and length (arXiv versus the
    published version), so only surnames are compared.
    """
    a = [surname(n) for n in found if norm(n) != "et al"]
    b = [surname(n) for n in mine if norm(n) != "et al"]
    short, long = sorted((a, b), key=len)
    return bool(short) and a[0] == b[0] and set(short) <= set(long)


def _compare(key: str, ident: str, meta: dict[str, Any], found: tuple) -> list[str]:
    title, authors, year = found
    official = str((meta.get("aliases") or [meta.get("title", "")])[0])
    problems = []
    if norm(title) != norm(official):
        problems.append(f"{key} {ident}: title {title!r} != {official!r}")
    if not same_authors(authors, [str(a) for a in meta.get("authors") or []]):
        problems.append(f"{key} {ident}: authors {authors} != {meta.get('authors')}")
    if key == "doi" and year != meta.get("year"):
        problems.append(f"{key} {ident}: year {year} != {meta.get('year')}")
    return problems


STOP = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with"}


def check_url(meta: dict[str, Any]) -> list[str]:
    """Check a note without doi/arxiv: ``url`` answers 200 and names the work."""
    url = str(meta.get("url") or "")
    if not url:
        return ["no doi, arxiv id or url"]
    try:
        page = get(url).decode("utf-8", "replace")
    except (OSError, LookupError) as exc:
        return [f"url {url}: not reachable ({exc})"]
    found = re.search(r"<title[^>]*>(.*?)</title>", page, re.IGNORECASE | re.DOTALL)
    haystack = html.unescape(found.group(1)) if found else ""
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname and parsed.hostname.endswith("huggingface.co"):
        haystack += " " + parsed.path
    have = set(norm(haystack).split())
    names = [str(n) for n in [*(meta.get("aliases") or []), meta.get("title", "")]]
    for name in names:
        words = set(norm(name).split()) - STOP
        if words and words <= have:
            return []
    return [f"url {url}: none of {names} found in page title {haystack.strip()!r}"]


def check_verified(meta: dict[str, Any]) -> list[str]:
    """Check the ``verified`` field: present, ``YYYY-MM-DD`` and not in the future."""
    raw = meta.get("verified")
    if not raw:
        return ["verified: missing"]
    try:
        day = raw if isinstance(raw, dt.date) else dt.date.fromisoformat(str(raw))
    except ValueError:
        return [f"verified: {raw!r} is not an ISO date YYYY-MM-DD"]
    return [f"verified: {raw} is in the future"] if day > dt.date.today() else []


def url_only(meta: dict[str, Any]) -> bool:
    """Tell whether a note has neither ``doi`` nor ``arxiv`` (checked by ``url``)."""
    return not any(meta.get(key) for key in SOURCES)


def check(path: Path) -> list[str]:
    """Return the problems with one note's citation (empty when it checks out)."""
    meta = frontmatter(path)
    problems = check_verified(meta)
    if url_only(meta):
        return problems + check_url(meta)
    for key, lookup in SOURCES.items():
        ident = str(meta.get(key) or "")
        if not ident or ident.lower().startswith("10.48550/"):
            continue  # arXiv DOIs are checked through the arxiv id
        try:
            found = lookup(ident)
        except (OSError, LookupError, ValueError, KeyError, ET.ParseError) as exc:
            problems.append(f"{key} {ident}: lookup failed ({exc})")
            continue
        problems += _compare(key, ident, meta, found)
    return problems


def paper_notes(research: Path) -> list[Path]:
    """Return the notes under ``research`` tagged ``type/paper``."""
    return [
        path
        for path in sorted(research.rglob("*.md"))
        if PAPER_TAG in (frontmatter(path).get("tags") or [])
    ]


def main(argv: list[str] | None = None) -> int:
    """Check the given notes (default: all paper notes); print one line per note."""
    args = sys.argv[1:] if argv is None else argv
    notes = [Path(a) for a in args] or paper_notes(ROOT / "research")
    failed = 0
    for note in notes:
        problems = check(note)
        failed += bool(problems)
        weak = " (url only)" if url_only(frontmatter(note)) else ""
        print(f"{'FAIL' if problems else 'OK' + weak} {note.name}")
        for problem in problems:
            print(f"  - {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
