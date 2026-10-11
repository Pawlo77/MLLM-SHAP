---
name: python-style
description: >-
  Use in Codex or when the .cursor/rules Python rules are not loaded, when writing or
  reviewing Python in src/, tests/, scripts/ or examples/.
---

# mllm-shap Python style

Read the rules that apply; they are the single source:

- `.cursor/rules/python-types.mdc`: imports, annotations, typed payloads, mypy strict.
- `.cursor/rules/python-docstrings.mdc`: Google-style docstrings and traceability tags.
- `.cursor/rules/python-logging.mdc`: stdlib logging, levels, lazy formatting.
- `.cursor/rules/minimal-diff.mdc`, `tdd.mdc` and `brevity.mdc`: scope, tests first, size budgets.

Where the `astral` plugin's uv, ruff or ty skills differ, the rules win: mypy strict (not
ty) gates prek, `make` targets wrap `uv`, and dependencies change in `pyproject.toml`
plus `uv lock`, never `pip install`.
