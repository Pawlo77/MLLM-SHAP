---
name: nlp-shap-notebook
description: >-
  Create or update an examples/ Jupyter notebook for a public nlp_shap workflow.
  Use when adding a notebook, cataloging examples, or running make notebooks.
paths:
  - examples/**
---

# nlp-shap notebook

## Scaffold (optional)

```bash
uv run python .cursor/skills/nlp-shap-notebook/scripts/new_notebook.py <stem> --dry-run
uv run python .cursor/skills/nlp-shap-notebook/scripts/new_notebook.py <stem> --title "My Workflow"
```

## Checklist

Follow `.cursor/rules/examples.mdc`:

1. One workflow per file; focused name
2. Cell 1 = all imports (stdlib/third-party, then `nlp_shap`)
3. Cover full public usability scope; no stubs
4. Dual-audience markdown (technical + business)
5. Catalog in `examples/README.md`, `docs/examples.rst`, and `README.md`
6. Cross-link from matching `docs/guides/<topic>.rst` when relevant
7. Run `make notebooks` — commit only with cell outputs

Pair with `tests/` (correctness) and `tests/benchmarks/` when hot path (`benchmarks.mdc`).
