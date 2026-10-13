---
paths:
  - "**/*.py"
  - "**/*.toml"
  - "Makefile"
---

<!-- Generated from .cursor/rules/uv-workflow.mdc by scripts/sync_rules.py; do not edit. -->

# uv workflow

- Install: `make install`; run tools with `uv run`.
- Dependencies: edit `pyproject.toml`, then `uv lock`; commit `uv.lock` with them. Never `pip install`.
- Hooks: `prek`, never the `pre-commit` CLI; `make prek-all` matches the remote Prek Checks.
- Python version: see `.python-version`.
