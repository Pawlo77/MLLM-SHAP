---
name: literature-search
description: >-
  Use when the user wants a whole research area covered, not a few papers: survey a
  domain, check whether research/ is complete, re-run or extend a past search, or answer
  "is the search systematic?".
---

# Literature search

A domain search is complete only when it is reproducible: a written protocol, every
source queried, counts at every stage, a full-text note for each kept paper and a record
of what was screened out. One paper at a time is skill `paper-notes`.

## 1. Protocol (before any query)

- Question and scope in one sentence each; agree them with the user.
- Inclusion and exclusion criteria; the project's own papers are always excluded.
- Copy `research/protocols/example.yaml` to `research/protocols/<domain>.yaml`: 15–20
  OpenAlex keyword queries, about 12 arXiv field queries, 15–20 seed papers for forward
  citations (central notes already in `research/`, or classics of a new area), the
  `since` date, include groups (method AND domain) and excludes. Query design:
  [reference.md](reference.md).

## 2. Harvest and screen

```bash
uv run python .agents/skills/literature-search/scripts/harvest.py research/protocols/<domain>.yaml --out outputs/literature/<domain>
```

- Fetch takes 10–30 minutes (rate limits): run it in the background, never with
  foreground sleeps. `--stage screen` re-screens without refetching.
- Read `candidates.csv` title and abstract by hand; fill `tier` (A core, B narrower,
  C survey or position), `decision` and `reason`. Keyword screens let through many false
  positives; the human pass decides.
- Report retrieved, deduplicated, kept and per-tier counts to the user with the tier
  rules, and agree which tiers get notes before writing any.

## 3. Read and write

- Every noted paper is read in full from its `fulltext/` snapshot (skill `paper-notes`;
  fallbacks in [reference.md](reference.md)); a paper without accessible full text, or a
  position paper with no method, is recorded, not noted.
- Check that a candidate is not already noted under another title (renamed arXiv
  versions are common).
- Write notes with skill `paper-notes`, batched by destination folder; add each to its
  folder table as you go, or `vault-check` flags isolated notes.
- New area or theme: its folder and folder note, a row in the parent's table, and its slug
  and placement row in `paper-notes/reference.md`.

## 4. Record

- Add an `outcome` block to the protocol YAML: counts per stage, what was noted, and
  every screened paper without a note with its reason. Search records live in
  `research/protocols/`, never as notes.
- Raw output stays in a git-ignored folder (`outputs/`).

## 5. Check and ship

```bash
uv run python .agents/skills/paper-notes/scripts/verify_citation.py
make vault-check
```

- Run the citation check in the background; fix each mismatch it reports (pitfalls in
  [reference.md](reference.md)).
- The vault hook checks the whole tree, so notes, tables and the protocol land in one
  commit; skill `ship`.
