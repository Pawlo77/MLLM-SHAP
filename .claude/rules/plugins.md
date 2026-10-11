---
paths:
  - "pyproject.toml"
  - "src/mllm_shap/plugins/**"
  - "src/mllm_shap/backends/**"
  - "src/mllm_shap/estimation/**"
  - "src/mllm_shap/value/**"
  - "src/mllm_shap/viz/**"
  - "src/mllm_shap/masking/**"
---

<!-- Generated from .cursor/rules/plugins.mdc by scripts/sync_rules.py; do not edit. -->

# Plugins

Procedure for a new estimator, estimand, backend, value_fn, normalizer, partition, absence_policy or renderer: skill `plugin`. The contract:

- The class takes `context: PluginContext | None = None` first and reads its settings from `context.config` and `context.option(key, default)` (options come from `explanation.plugin_options[<name>]`).
- It is registered in `[project.entry-points."mllm_shap.<group>"]` and in `BUILTIN_PLUGINS` (`src/mllm_shap/plugins/builtin.py`); a test checks they agree.
- Plugin modules import `PluginContext` only under `TYPE_CHECKING` (a runtime import creates an import cycle).
- Heavy dependencies go in extras (`lmstudio`, `transformers`, `api`, `viz`).
- Tests resolve it through `PluginRegistry` and check real behaviour; add a scenario to `tests/practical/scenarios.yaml` (`tests/practical/test_catalog.py` checks the estimation, value, masking and partition groups).
- No empty plugin packages or unregistered public kinds.
