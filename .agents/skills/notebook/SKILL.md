---
name: notebook
description: >-
  Use when adding or updating an examples/ notebook, cataloging examples, or running
  make notebooks.
paths:
  - examples/**
---

# mllm-shap notebook

## Scaffold (optional)

```bash
uv run python .agents/skills/notebook/scripts/new_notebook.py <stem> --dry-run
uv run python .agents/skills/notebook/scripts/new_notebook.py <stem> --title "My Workflow"
```

Examples are notebooks only: edit the `.ipynb` directly and run `make notebooks` so the
committed file carries its outputs; `tests/examples/test_catalog.py` fails for a notebook
without outputs or for a stray `.py` file under `examples/`.

## Checklist

Follow `.cursor/rules/examples.mdc`:

1. One workflow per file; focused name
2. Cell 1 = all imports (stdlib/third-party, then `mllm_shap`)
3. Cover full public usability scope; no stubs
4. Dual-audience markdown (technical + business)
5. Catalog in `examples/README.md` and `docs/examples.rst`
6. Cross-link from matching `docs/guides/<topic>.rst` when relevant
7. Run `make notebooks` (executes every notebook in place; no sync step); commit only with
   cell outputs. Run it also after a `src/` change to a public workflow and before a release PR

Pair with `tests/` (correctness) and `tests/benchmarks/` when hot path (`benchmarks.mdc`).
