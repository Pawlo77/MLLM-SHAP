---
paths:
  - ".github/**"
---

<!-- Generated from .cursor/rules/ci.mdc by scripts/sync_rules.py; do not edit. -->

# CI

<<<<<<< before updating
## Workflows

| File | Trigger | Purpose |
|------|---------|---------|
| `prek.yaml` | push/PR to `main` | job `prek`: `prek run --all-files`, then `check_vault.py --strict --strict-tags` (full history) |
| `test.yaml` | push/PR to `main` | matrix `test` (3.12-3.14, coverage 85%, Codecov; `slow` on 3.12), `notebooks`, summary `tests-passed` |
| `codeql.yaml` | push/PR, weekly | job `CodeQL`; skipped while the repository is private |
| `deploy-docs.yaml` | push to `main` (docs, examples, knowledge, src), manual | `make -C docs html`, GitHub Pages artifact |
| `benchmarks.yaml` | weekly, manual | `make bench`, JSON report as an artifact (informational) |
| `pr-title.yaml` | PR events | job `title` (required): `type: summary`; drafts and WIP fail |
| `assign-author.yaml` | PR opened, reopened, title edited | assignee and label from the title type |
| skill `ship` (local) | finished branch | opens the PR with Copilot review, fixes its threads; routine changes (`scripts/pr_policy.py`) merge themselves, others wait for the maintainer |
| `publish.yaml` | tag `v*.*.*`, pushed by skill `release` after the merge | tag must equal `uv version --short`; build, smoke test, `uv publish`, GitHub Release; `make upgrade` never edits it |

Required on `main`: `prek`, `tests-passed`, `CodeQL`, `title`; keep job names stable. Repository settings and the branch-protection command: `.agents/skills/ci/reference.md`.
=======
The workflows of this repository, their triggers and the required checks:
`ci-workflows.mdc`. Settings and branch protection: `.agents/skills/ci/reference.md`.
>>>>>>> after updating

## Conventions

- Top-level `permissions: contents: read`; a job widens only what it needs.
- Every `actions/checkout` sets `persist-credentials: false`.
- Pin actions by exact version tag; the `zizmor` prek hook (`.github/zizmor.yml`) reports zero findings.
- uv `version: "0.12.23"` in every workflow, equal to `required-version` in `pyproject.toml`.
- Pass `${{ ... }}` into `run:` through `env:`, never inline.
- No `paths:` filter on a workflow with a required check: a skipped workflow never reports.
<<<<<<< before updating
- The test matrix sets `UV_PYTHON`, because `uv run` honours `.python-version`.
- The notebook comparison step is advisory (`continue-on-error`); the execute step blocks.
- Secrets: only the optional `CODECOV_TOKEN`; PyPI uses trusted publishing.
=======
- Keep job names stable: branch protection requires them by name.
>>>>>>> after updating
