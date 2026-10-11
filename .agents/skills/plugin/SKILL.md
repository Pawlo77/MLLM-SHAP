---
name: plugin
description: >-
  Use when adding a plugin, entry point or PluginRegistry kind (estimator, estimand,
  backend, value_fn, normalizer, partition, absence_policy, renderer).
paths:
  - pyproject.toml
  - src/mllm_shap/plugins/**
  - src/mllm_shap/backends/**
  - src/mllm_shap/estimation/**
  - src/mllm_shap/value/**
  - src/mllm_shap/viz/**
  - src/mllm_shap/masking/**
---

# mllm-shap plugin

## Scaffold (optional)

```bash
uv run python .agents/skills/plugin/scripts/new_plugin.py <group> <name> --dry-run
uv run python .agents/skills/plugin/scripts/new_plugin.py <group> <name>
```

Creates a stub module (constructor `__init__(self, context: "PluginContext | None" = None)`,
traceability tags), a test that resolves it through `PluginRegistry` plus a failing behaviour
test, and the pyproject entry point. Then implement for real.

## Steps

The plugin contract (constructor, `BUILTIN_PLUGINS`, `TYPE_CHECKING` import, extras) is
`.cursor/rules/plugins.mdc`; read it first.

1. Implement the typed class under the matching package (`estimation/`, `backends/`, `value/`, `viz/`, `masking/`).
2. Register it in `pyproject.toml` (`[project.entry-points."mllm_shap.<group>"]`) and in `BUILTIN_PLUGINS`.
3. **Red**: test resolution through `PluginRegistry` and a behaviour test (not only import);
   add a scenario to `tests/practical/scenarios.yaml` and run `/practical <id>`.
4. **Green**: the smallest implementation that passes.
5. If user-facing: notebook (`examples.mdc`) and docs (`docs.mdc`); a new backend always needs
   a benchmark and a minimal notebook.
6. Update `knowledge/architecture/<pkg>.md` for every changed package (skill `notes`) and run
   `make knowledge`, then the exit criteria in `AGENTS.md`.

Groups: `estimators`, `estimands`, `partitions`, `absence_policies`, `value_fns`,
`normalizers`, `backends`, `renderers`.
