# Paper notes: writing standard

This project's half of skill `paper-notes`; edit it freely. Every note has the same
sections, words and symbols.

## Layout

```text
research/
  Research.md                  hub: the question and a table of areas
  AGENTS.md                    agent context (tooling, project use)
  protocols/<domain>.yaml      literature-search protocols and outcomes
  <Area>/
    <Area>.md                  folder note: table of its themes or papers
    <Short title>.md           paper note, citekey in frontmatter
    <Theme>/
      <Theme>.md               folder note of the theme
      <Short title>.md
```

File names are the paper's short name (method or system name), never `Year - Author - Title`.

## Placement

| Paper changes | Folder | Tag |
|---------------|--------|-----|
| (what a paper must change to belong here) | `research/<Area>/[<Theme>/]` | `theme/<slug>` |

## Sections (exact order, no others)

| Section | Contract |
|---------|----------|
| `> [!info] TL;DR` | What the paper changes and its main result, 3–5 plain sentences |
| `## Motivation` | The gap before the paper, 3–6 bullets; prior work linked (vault note or verified URL) |
| `## Method` | Notation sentence; every definition, equation, algorithm step and theorem needed to re-implement or check the paper, in LaTeX (`$…$`, `$$…$$`); an `Intuition:` sentence after each key equation; theorems with their assumptions |
| `## Implementation` | Code (URL or "not released"), software and frameworks, hardware, architectures and sizes, neuron and surrogate parameters, time steps, optimiser and schedule, training budget |
| `## Data` | One bullet per dataset: name, what it is, size, splits, encoding or preprocessing |
| `## Experiments` | One bullet per experiment: `**Name:** setup, baselines and metrics; what was found` |
| `## Results` | The main and ablation results in numbers, each with setting and baseline; described in prose, never a copied table |
| `## Limitations` | One bullet per failure mode, marked `(authors)` when the paper states it, `(ours)` otherwise |
| `## Relation to our project` | What we take, test or avoid; links to project notes |
| `## Future work` | Bullets led by `**Done:**` or `**Partly done:**` (with a verified link) or `**Open:**` ending `(searched YYYY-MM)` |

Whole note: 1200–3500 words; sentences of at most 35 words. A section the paper has nothing for says so in one line (for example, a theory paper's `## Data`).

## Full-text snapshots

`fulltext/` mirrors `research/` (`fulltext/<Area>/[<Theme>/]<Short title>.txt`): the 1:1 extracted text of each paper, git-ignored and hidden in Obsidian. `scripts/snapshot.py` creates it from the note's `arxiv`, `doi` or `url`. Keep it in step with `research/`: `--sync` after moving a note, `--missing` to list gaps, `--all` to rebuild on a fresh clone. When a supplement, appendix or second version holds method or results, put its URL in the note's `supplement` field so the text is appended and survives a rebuild. Later passes read the snapshot instead of downloading again.

## Review

An independent reviewer (not the writer) checks every claim, number and equation against the snapshot, looks up each future-work link, adds what is missing for the contract, and sets `reviewed` to the date. A claim from a supplement or released code that the snapshot lacks is looked up there before it is deleted. A note without `reviewed` is a draft.

## Never

- References to figures, tables or equation numbers: state the finding.
- Copied result tables; abstract paraphrase; "this paper proposes"; TODOs.
- Facts not in the paper or a verified source; citations from memory.
- The same point in two sections.

## Glossary

| Use | Not | Meaning |
|-----|-----|---------|
| (term) | (synonyms to avoid) | (meaning) |

## Notation

(The project's symbols. Rename a paper's symbols to these and say so once.)

## Tags (vocabulary)

Paper notes carry `type/paper`, one `theme/` and one or more `topic/`:

- `theme/`: (one slug per row of Placement).
- `topic/`: (add values here before using them).

Other notes: `type/hub`, `type/theme`.

## Citation fields

`title` (short name = filename = heading), `aliases` (official title first), `authors`
(full names, published order), `year`, `venue` (with volume and pages), `arxiv` and/or
`doi`, `url` (published version, else arXiv), `citekey`, `verified` (`YYYY-MM-DD` the citation and full text were checked; there is no Verification section)
(`<firstauthor><year><firstword>`), `theme`, `tags`. `scripts/verify_citation.py` must
print `OK` for the note.
