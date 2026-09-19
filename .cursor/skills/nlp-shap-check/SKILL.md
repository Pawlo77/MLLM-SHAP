---
name: nlp-shap-check
description: >-
  Validate nlp-shap changes before hand-off. Use when finishing a task, before
  commit/push, or when the user asks to check, verify, or run make check.
---

# nlp-shap check

## Default validate

```bash
make agent-check
```

Same as `make check`: filtered pytest (`not lms and not gpu and not bench`) + `prek-all`.

Optional full marker run: `make tests-all`.

## Conditional

| Changed | Also run |
|---------|----------|
| `docs/` or public API | `make docs` |
| `examples/*.ipynb` | `make notebooks` |
| deps / `pyproject.toml` | `uv lock` then `make agent-check` |
| hot path / backend | `make bench` (local; not CI) |

## After push

```bash
gh run list --branch "$(git branch --show-current)" --limit 5
gh run watch <run-id> --exit-status
```

## Stop

Do not commit/push/PR unless user asked (`git-commits.mdc`).
