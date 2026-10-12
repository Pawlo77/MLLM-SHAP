---
name: ci
description: >-
  Use after a push, when CI is red, when editing .github/workflows, or when asked about
  checks, branch protection or repository settings.
---

# CI

## Green gate for every pull request

A pull request is not finished until `main` is merged into it and every check is green:
run `uv run python scripts/pr_follow.py` after each push until it exits 0.

1. Before each push: `git fetch`; if `git log HEAD..origin/main` lists commits, merge
   `origin/main` into the branch (no rebase, no force push), rerun `make check`, resolve
   conflicts. `gh pr view <n> --json mergeStateStatus` must not say `BEHIND` or `DIRTY`.
2. After each push: watch the PR's checks (desktop app: `ccd_pr` `get_status`; else
   `gh pr checks <n>`) until the required ones pass; a skipped CodeQL is fine.
3. A red check: read `gh run view <run-id> --log-failed`, reproduce locally, push the fix
   to the same branch, and watch again. Report the PR only when it is green and up to date.
4. Reproduce CI, not your machine: tests must not need git-ignored or local-only files
   (plugin `main.js`, `.venv` extras, `mlflow.db`, tokens); check a clean state when a test
   touches such files.

## After every push

```bash
gh run list --branch "$(git branch --show-current)" --limit 5
gh run watch <run-id> --exit-status        # once per workflow run
```

A red run is fixed on a follow-up commit on the same branch. Never leave `main` red.

## Reading a failure

```bash
gh run view <run-id>                       # jobs and their status
gh run view <run-id> --log-failed          # only the failing steps' logs
gh run view <run-id> --job <job-id> --log  # one job in full
```

Reproduce locally before pushing a fix: `make check` covers Prek Checks and Pytest; other
jobs and their local commands: `local.md` in this folder, when it exists.

## Workflows and settings

Workflow table and conventions: `.cursor/rules/ci.mdc` and `ci-workflows.mdc`. Repository
settings and branch protection (applied by the maintainer): [reference.md](reference.md).

Before pushing a workflow edit: `uvx zizmor --min-severity=medium .github/workflows`
reports 0 findings (it also runs in prek).
