"""Keep a 1:1 full-text snapshot of each paper note under ``fulltext/``.

Usage: ``uv run --with pypdf python <this script> NOTE [--pdf SOURCE] [--force]`` writes
``fulltext/<path of NOTE relative to research/>.txt``. The source is ``--pdf`` (a URL or
local file), else the arXiv PDF of the note's ``arxiv`` field, else its ``url``. Each
URL in the ``supplement`` field is appended after a ``# supplement:`` line.
``--all`` writes every missing snapshot (a fresh clone has none); ``--missing`` lists
paper notes without one; ``--sync`` moves snapshots whose note moved (same file name,
one match) and lists orphans. ``fulltext/`` is git-ignored and hidden in Obsidian: the
snapshots are working copies for reading, never committed.
"""

import argparse
import datetime as dt
import html
import io
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
"""Repository root, four levels above this script."""

RESEARCH = ROOT / "research"
FULLTEXT = ROOT / "fulltext"
PAPER_TAG = "type/paper"
AGENT = "Mozilla/5.0 (paper-notes snapshot)"


def frontmatter(path: Path) -> dict:
    """Return the YAML frontmatter of a note, or an empty dict."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    return yaml.safe_load(text.split("---\n", 2)[1]) or {}


def paper_notes() -> list[Path]:
    """Return every note under ``research/`` tagged as a paper."""
    return sorted(
        p
        for p in RESEARCH.rglob("*.md")
        if PAPER_TAG in (frontmatter(p).get("tags") or [])
    )


def target(note: Path) -> Path:
    """Return the snapshot path that mirrors ``note``."""
    return FULLTEXT / note.resolve().relative_to(RESEARCH).with_suffix(".txt")


def fetch(url: str) -> tuple[bytes, str]:
    """Download ``url`` with retries, or read a local file; return body and type."""
    if Path(url).is_file():
        return Path(url).read_bytes(), ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=90) as resp:  # nosec B310 - http(s) only
                return resp.read(), resp.headers.get("Content-Type", "")
        except OSError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def pdf_text(body: bytes) -> str:
    """Extract the text of a PDF, one ``=== p<N>`` marker per page."""
    from pypdf import PdfReader  # imported here: only needed for PDFs

    pages = PdfReader(io.BytesIO(body)).pages
    return "\n".join(
        f"=== p{i}\n{p.extract_text() or ''}" for i, p in enumerate(pages, 1)
    )


def html_text(body: bytes) -> str:
    """Reduce an HTML page to readable text."""
    page = body.decode("utf-8", errors="replace")
    page = re.sub(
        r"<(script|style|svg|nav|header|footer)[^>]*>.*?</\1>",
        " ",
        page,
        flags=re.DOTALL | re.IGNORECASE,
    )
    page = re.sub(
        r"<math[^>]*alttext=\"([^\"]*)\"[^>]*>.*?</math>",
        r" $\1$ ",
        page,
        flags=re.DOTALL,
    )
    page = re.sub(r"</(p|div|h[1-6]|li|tr|section)>", "\n", page, flags=re.IGNORECASE)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def sources(meta: dict, pdf: str | None) -> list[str]:
    """Return candidate URLs in order of preference."""
    urls = [pdf] if pdf else []
    if meta.get("arxiv"):
        urls.append(f"https://arxiv.org/pdf/{meta['arxiv']}")
    if meta.get("url"):
        urls.append(str(meta["url"]))
    return urls


def decode(body: bytes, kind: str) -> str:
    """Return the text of a PDF, a plain-text file or an HTML page."""
    if body[:5] == b"%PDF-":
        return pdf_text(body)
    if kind.startswith("text/plain"):
        return body.decode("utf-8", errors="replace")
    return html_text(body)


def extract(url: str) -> str:
    """Download ``url`` and return its text."""
    return decode(*fetch(url))


def supplement_text(meta: dict) -> str:
    """Return the text of the note's ``supplement`` URLs, each under its own line."""
    urls = meta.get("supplement") or []
    text = ""
    for url in [urls] if isinstance(urls, str) else urls:
        text += f"\n\n# supplement: {url}\n{extract(str(url))}"
    return text


def snapshot(note: Path, pdf: str | None, force: bool) -> str:
    """Write the snapshot of one note; return a one-line status."""
    out = target(note)
    rel = note.resolve().relative_to(ROOT)
    if out.exists() and not force:
        return f"exists {out.relative_to(ROOT)}"
    meta = frontmatter(note)
    for url in sources(meta, pdf):
        try:
            body, kind = fetch(url)
        except OSError as err:
            print(f"  {url}: {err}", file=sys.stderr)
            continue
        text = decode(body, kind)
        if len(text.split()) < 1500:
            print(
                f"  {url}: only {len(text.split())} words ({kind}), skipped",
                file=sys.stderr,
            )
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        today = dt.date.today().isoformat()
        head = f"# source: {url}\n# retrieved: {today}\n# note: {rel}\n\n"
        out.write_text(head + text + supplement_text(meta), encoding="utf-8")
        return f"wrote {out.relative_to(ROOT)} ({len(text.split())} words)"
    return f"FAIL {rel}: no readable full text; pass --pdf URL"


def sync() -> None:
    """Move snapshots whose note moved; print orphans."""
    by_name: dict[str, list[Path]] = {}
    for note in paper_notes():
        by_name.setdefault(note.stem, []).append(note)
    for txt in sorted(FULLTEXT.rglob("*.txt")):
        notes = by_name.get(txt.stem, [])
        if len(notes) == 1 and target(notes[0]) != txt:
            dest = target(notes[0])
            dest.parent.mkdir(parents=True, exist_ok=True)
            txt.rename(dest)
            print(f"moved {txt.relative_to(ROOT)} -> {dest.relative_to(ROOT)}")
        elif not notes:
            print(f"orphan {txt.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notes", nargs="*", type=Path)
    parser.add_argument(
        "--pdf", help="PDF or HTML URL, or a local file, for a single note"
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing snapshot"
    )
    parser.add_argument(
        "--all", action="store_true", help="write every missing snapshot"
    )
    parser.add_argument(
        "--missing", action="store_true", help="list notes without a snapshot"
    )
    parser.add_argument(
        "--sync", action="store_true", help="move snapshots of moved notes"
    )
    args = parser.parse_args(argv)
    if args.sync:
        sync()
    if args.missing:
        for note in paper_notes():
            if not target(note).exists():
                print(note.relative_to(ROOT))
    failed = False
    todo = [n for n in paper_notes() if not target(n).exists()] if args.all else []
    for note in [*args.notes, *todo]:
        status = snapshot(note, args.pdf, args.force)
        failed |= status.startswith("FAIL")
        print(status)
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
