# Research agents

Literature notes for human readers (Obsidian vault at the repository root); notes hold no
tooling, templating or plans, that context lives here and in skill `paper-notes`.

- Layout: hub [[Research]]; paper notes at `<Area>/[<Theme>/]<Short title>.md` with a `citekey`; every folder has a note of its own name with a table of its contents; search protocols in `protocols/`; no PDFs; full-text snapshots in git-ignored `fulltext/`, mirroring `research/` (tree in `.agents/skills/paper-notes/reference.md`).
- Folder notes (area and theme): scope line, `## Summary` (3–5 bullets synthesising the papers, with links), the themes table (`| Theme | Papers | What it covers |`) and/or the papers table, then `## Future work` (open items merged from the paper notes, each linking its source notes).
- Fields, sections, tags and themes: `.agents/skills/paper-notes/reference.md`; new notes start from `template.md` beside it.
- New papers: skill `paper-notes` (citation verified first); covering an area: skill `literature-search` (protocol and outcome in `protocols/`).
- Project use (which method the code implements, config keys): record it here, not in the notes.
- After edits: `make vault-check`.
