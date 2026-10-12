---
name: check
description: >-
  Use when asked to check, verify or run make check, and before hand-off.
---

<<<<<<< before updating
# mllm-shap check
=======
# Check
>>>>>>> after updating

## Default validate

```bash
make check
```

<<<<<<< before updating
`make check` runs `make agent-check` (filtered pytest with
`not lms and not gpu and not bench and not slow and not practical`, then `prek-all`), then
`make vault-check`.

Optional full marker run: `make tests-all`.

## Conditional

| Changed | Also run |
|---------|----------|
| `docs/` or public API | `make docs` |
| `examples/*.ipynb`, a `src/` change to a public workflow, or a release PR | `make notebooks` |
| dependencies or `pyproject.toml` | `uv lock`, then `make check` |
| hot path or backend | `make bench` (local); `make bench-regression` against `tests/benchmarks/baseline.json` |
| `src/mllm_shap/<pkg>/`, docstrings, `knowledge/`, `plans/` | `make knowledge vault-check`; architecture note updated (skill `notes`) |

## After push

```bash
gh run list --branch "$(git branch --show-current)" --limit 5
gh run watch <run-id> --exit-status
```
=======
`make check` runs the tests, every prek hook and the vault check. What each part covers
here, and what else to run for the files you changed: `local.md` in this folder.

## After push

Run `uv run python scripts/pr_follow.py` until it exits 0 (rule `git-commits`, skill `ci`).
>>>>>>> after updating

## Stop

Do not commit, push or open a PR unless the user asked (`git-commits.mdc`).
