---
name: notes
description: >-
  Use when writing or updating a vault note in knowledge/: the architecture note after a
  src/ change, a decision-log or papers section, or an issue, concept, handbook or
  solution note.
---

# mllm-shap notes

Notes live in `knowledge/` (vault rules: `knowledge/README.md`; machinery: skill
`obsidian`). A note is updated in the same change as the code it describes.

## Where and how

- Paths, frontmatter keys and vault rules: `knowledge/README.md`. Sections and
  type-specific keys: the template `knowledge/_templates/<type>.md` (`module`, `issue`,
  `concept`, `handbook`, `solution`, `plan`); `make vault-check` enforces them.
- A decision is a section appended to `knowledge/decisions/<Domain> decisions.md`, newest
  first (`_templates/decision.md`); a paper is a section of `knowledge/papers/Papers.md`
  (`_templates/paper.md`). Link one with `[[<Log>#<heading>|<title>]]`.
- Link a new note from its parent: the map `knowledge/mllm-shap.md` (architecture, concept,
  handbook, log), the architecture note's Known issues (issue),
  `knowledge/handbook/Solutions/README.md` (solution); link a decision from its cause.
- Never edit `knowledge/generated/`; `make knowledge` rebuilds it.

## Writing rules (beyond `knowledge/README.md`)

- Every sentence carries a mechanism, a number, an invariant or a decision; no process
  history or status chatter.
- Code names in backticks. Describe the code as it is; planned behaviour is "changes in
  <version>:" with a link to the issue or decision.
- Papers: DOI or arXiv links only, no page, table, figure or equation numbers; the
  Claim is one sentence on what the library takes from the paper.

## Procedure: update the architecture note after a `src/` change

1. `git diff --stat main...HEAD -- src/mllm_shap/<pkg>` and read the changed modules.
2. Diff the public surface: names added, removed, renamed, signatures changed
   (`make knowledge`, then read `knowledge/generated/Public API.md` and `Modules.md`).
3. Rewrite "Public surface" and "Data flow" for what changed; adjust "Invariants".
4. Update "Known issues": link new issue notes; mark fixed ones in the issue note
   (`status: fixed`, `fixed_in: <version>`) and drop "changes in <version>" wording that
   has landed.
5. Add new modules to `related-code` and new test files to "Tests"; remove deleted ones.
6. Bump `updated:` to today.
7. Append a decision section for any design choice the change made.

## Procedure: issue found or fixed

- Found: create the issue note (`status: open`), link it from the architecture note's
  Known issues, cite the repro you ran.
- Fixed: `status: fixed`, `fixed_in: <version>`, link the plan task or pull request in
  "Fix"; keep Symptom, Cause and Repro as history.

## Procedure: solution note (vibe step 9, after the PR merges)

Copy `_templates/solution.md` to `knowledge/handbook/Solutions/<YYYY-MM-DD> - <title>.md`
(the title names the problem, not the fix) and fill every section: each failed attempt with
the evidence that it failed, the change with file and function, and one imperative reusable
rule. `related-code`, `pr` and `plan-task` are filled in.

## Procedure: paper

1. Find candidates with the `paper-scout` agent (Semantic Scholar and arXiv, verified
   DOI or arXiv URLs, deduplicated against `knowledge/papers/Papers.md`); the user picks.
2. Add its section to `Papers.md` (`_templates/paper.md`).
3. For a paper whose method the library implements, run the `methodologist` agent for
   its "Paper-to-code compliance" table (each equation and the function that
   implements it, or "not implemented"); the Claim states exactly what the table shows.
4. `note-fact-checker` checks the section against the paper and the compliance table.

## After writing

```bash
make knowledge vault-check
```

Then run the `note-fact-checker` agent on each new or substantially changed
architecture or concept note, decision or paper section, and fix every confirmed mismatch.
