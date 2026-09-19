---
name: nlp-shap-plugin
description: >-
  Register a new nlp_shap plugin entry point (estimator, estimand, backend,
  value_fn, normalizer, partition, absence_policy, renderer). Use when adding
  a plugin, entry point, or PluginRegistry kind.
paths:
  - pyproject.toml
  - src/nlp_shap/plugins/**
  - src/nlp_shap/backends/**
  - src/nlp_shap/estimation/**
  - src/nlp_shap/value/**
  - src/nlp_shap/viz/**
  - src/nlp_shap/masking/**
---

# nlp-shap plugin

## Scaffold (optional)

```bash
uv run python .cursor/skills/nlp-shap-plugin/scripts/new_plugin.py <group> <name> --dry-run
uv run python .cursor/skills/nlp-shap-plugin/scripts/new_plugin.py <group> <name>
```

Creates stub module + failing test + pyproject entry point. Then implement for real.

## Steps

1. Implement the typed class under the matching package (`estimation/`, `backends/`, `value/`, `viz/`, `masking/`).
2. Add `[project.entry-points."nlp_shap.<group>"]` in `pyproject.toml` — see existing groups.
3. Wire any builtin defaults in `src/nlp_shap/plugins/` if the group expects them.
4. **Red** — test resolution via `PluginRegistry` + a behavior test (not only import).
5. **Green** — smallest impl that passes.
6. If user-facing: notebook (`examples.mdc`) + docs (`docs.mdc`).
7. `make agent-check` (+ `make docs` / `make notebooks` when applicable).

## Groups (current)

`estimators` · `estimands` · `partitions` · `absence_policies` · `value_fns` · `normalizers` · `backends` · `renderers`

## Rules

- Optional heavy deps → extras only (`lmstudio`, `transformers`, `api`, `viz`)
- Relative imports inside `src/nlp_shap/`
- No empty package scaffolds
