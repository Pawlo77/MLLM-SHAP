---
paths:
  - "papers/**/*.tex"
  - "papers/**/*.bib"
  - "papers/**/*.sty"
  - "papers/**/*.cls"
---

<!-- Generated from .agents/rules/latex-papers.mdc by scripts/sync_rules.py; do not edit. -->

# LaTeX papers

- Skill: `latex-papers`; the folder is `papers-dir` in `[tool.agent-kit]` of `pyproject.toml`.
- One source tree per paper; no parallel copies.
- Build junk stays git-ignored; commit a PDF only where the folder's `.gitignore` or `AGENTS.md` says so.
- Python TDD, docstring and ruff rules do not apply here.
