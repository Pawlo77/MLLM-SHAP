---
name: development
description: >-
  Use for any code, test, packaging, hook or CI change in this repository.
---

# Development

Exit criterion: `make check` (`AGENTS.md`). Rules: `.cursor/rules/*.mdc` (always-apply rules
plus the ones whose globs match the files you touch). The conventions, extra steps and
pitfalls of this repository: `local.md` in this folder, when it exists.

## Workflow

0. **Set up**: when session start says `setup: incomplete`, run `make setup` first and
   report its result.
1. **Read**: task scope only; the existing `src/` and `tests/`; what `local.md` names.
2. **Red**: failing `tests/test_*.py` first (mirror `src/`); test behaviour, not trivia.
3. **Green**: smallest typed implementation; reuse before new abstractions.
4. **Refactor**: simplify while green; no unrelated cleanup.
5. **Validate**: re-read the diff; `make check` (it includes `make vault-check`) and the
   checks `local.md` adds; fix every issue found.
6. **Hand off**: stop unless the user asks for a commit, push or PR (`git-commits.mdc`).

Packaging and dependencies: `pyproject.toml` plus `uv lock`. A new Makefile target gets a
`name: ## help` line and a `.PHONY` entry. After push: skill `ci`.

## Pitfalls

- `prek`, not `pre-commit`; `uv lock`, not `pip install`.
- Implementation before the test: go back to Red.
- A push without `make check` turns the remote checks red.
