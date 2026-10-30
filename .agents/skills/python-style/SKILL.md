---
name: python-style
description: >-
  Use in Codex or when the .cursor/rules Python rules are not loaded, when writing or
  reviewing Python in this repository.
---

# Python style

Read the rules that apply; they are the single source. `local.md`, when it exists, lists this repository's
Python rules (typing, docstrings, logging, configuration). `minimal-diff.mdc`, `tdd.mdc`
and `brevity.mdc` set scope, tests first and size budgets.

Where the `astral` plugin's uv, ruff or ty skills differ, the rules win: mypy strict (not
ty) gates prek, `make` targets wrap `uv`, and dependencies change in `pyproject.toml`
plus `uv lock`, never `pip install`.
