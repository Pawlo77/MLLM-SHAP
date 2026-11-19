---
name: paper-notes
description: >-
  Use when adding, summarizing or updating a literature note in research/, citing a
  paper from the vault, or finding a few papers on a topic for the project.
---

# Paper notes

One note per paper at `research/<Area>/[<Theme>/]<Short title>.md`, hub `research/Research.md`;
every folder has a note named after it that lists its papers. No PDFs are kept; the
extracted text of each paper lives in git-ignored `fulltext/`, mirroring `research/`. The
project's half of this skill is [reference.md](reference.md) (layout and placement,
section contracts, tags, glossary, notation); every note starts from [template.md](template.md).

## 1. Verify the paper exists (gate)

- Find the record yourself: arXiv API (`https://export.arxiv.org/api/query?id_list=<id>`),
  Crossref (`https://api.crossref.org/works/<doi>`), the proceedings page, Semantic
  Scholar for the published venue. Never cite from memory.
- Prefer the published version: its title, author order, venue with volume and pages, DOI.
- Fill the frontmatter, then run the checker; continue only on `OK`:
  ```bash
  uv run python .agents/skills/paper-notes/scripts/verify_citation.py "research/<Area>/<Short title>.md"
  ```
- Status `OK` means the DOI/arXiv record matches (Crossref, else DataCite) and `verified`
  is a past ISO date. `OK (url only)` is weaker, for sources without DOI or arXiv id
  (proceedings, model cards, GitHub, Zenodo): only that `url` resolves and the title words
  occur in its page title. Say so in the report; use it only when no id exists.
- Prior work linked in Motivation is checked the same way.
- Finding candidates for the user: agent `paper-scout`, then a table of citation,
  verified URL, one-line relevance and theme; the user picks. A whole area: skill
  `literature-search`.

## 2. Place

Pick the area and theme from the placement table in [reference.md](reference.md). None
fits: propose one to the user, then add its row and tag slug there, create the folder with
its folder note and add it to the parent's table (`Research.md` for an area).

## 3. Read

Snapshot the full text, then read all of it (appendices with method or results too):

```bash
uv run --with pypdf python .agents/skills/paper-notes/scripts/snapshot.py "research/<Area>/<Short title>.md"
```

Other sources: `--pdf <url or file>`. Upkeep flags and supplements: [reference.md](reference.md).
A paper without readable full text is recorded as unread, never written from the abstract.

## 4. Write

1. Fill every section of the template in order, following the contracts, glossary and
   notation in [reference.md](reference.md). Give equations (LaTeX), algorithms,
   theorems, software, hyperparameters and results: enough to check the work without the PDF.
2. Write for human readers: no tooling, plans or config keys in the note; project usage
   goes to `Relation to our project`, tooling to `research/AGENTS.md`.
3. Future work: the paper's own directions, then its main gaps; search citations and arXiv
   for each and mark it `Done`, `Partly done` (link the verified later paper) or
   `Open (searched YYYY-MM)`.
4. Re-read the source for every number, dataset and claim you wrote; delete what you
   cannot find. Then set `verified` in the frontmatter to today's date.
5. Re-read as a newcomer: each idea once.

## 5. Review (another agent or person)

Never the writer: the reviewer checks every claim, number, equation and future-work link
against the snapshot, fixes and completes the note, and sets `reviewed`.

## 6. Link

- Add the paper to the table of its folder note (an unlinked note fails `vault-check`).
- Link related vault notes from Motivation or Relation to our project, not a "See also".

## 7. Check

```bash
uv run python .agents/skills/paper-notes/scripts/verify_citation.py
make vault-check
```

Report the note, its theme and anything left out as unverifiable.
