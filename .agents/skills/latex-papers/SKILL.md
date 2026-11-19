---
name: latex-papers
description: >-
  Use when editing or building the project's LaTeX (papers, report, thesis): .tex, .bib,
  .sty or .cls files, compiling, fixing citations or bibliography entries.
---

# LaTeX papers

The LaTeX folder is `papers-dir` in `[tool.agent-kit]` of `pyproject.toml` (`report`
in the experiment scaffold, else `papers`). Read `local.md` in this folder, when it
exists, and the `AGENTS.md` of the paper's folder: they name the main files and the
build command.

## Rules

- Python rules (TDD, docstrings, ruff) do not apply to `.tex`.
- Edit the one source tree of each paper; never start a parallel copy.
- Build junk (`*.aux`, `*.log`, `*.synctex.gz`, ...) stays git-ignored. Commit a compiled
  PDF only where the folder's `.gitignore` or `AGENTS.md` says so, or the user asks.
- Keep bib keys stable; a cited paper with a note in `research/` uses its `citekey`.
- Numbers in the text come from executed output or a verified paper note, never memory.

## Workflow

1. Read the folder's `AGENTS.md` (or `local.md`) for the main file and build command;
   default `latexmk -pdf <main>.tex` inside the folder.
2. Edit minimally; match the existing macros, labels and citation style.
3. Build only when the user asks or to verify a compile fix; report errors and
   undefined references.
4. Hand off; commit only when asked.
