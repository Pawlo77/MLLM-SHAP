---
name: ci
description: >-
  Use after a push, when CI is red, when editing .github/workflows, or when asked about
  checks or branch protection.
---

# mllm-shap CI

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

Reproduce locally before pushing a fix: `make check` covers Prek Checks and Pytest;
`make docs` covers Deploy Docs; `make notebooks` covers the notebook job.
A matrix failure on one Python version only: `UV_PYTHON=3.14 uv run pytest ...`.

## Workflows and settings

Workflow table and conventions: `.cursor/rules/ci.mdc`. Repository settings and branch
protection (applied by the maintainer): [reference.md](reference.md).

Before pushing a workflow edit: `uvx zizmor --min-severity=medium .github/workflows`
reports 0 findings (it also runs in prek).
