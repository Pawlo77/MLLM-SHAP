---
title: "plugins"
type: module
domain: plugins
tags: [architecture, plugins]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/nlp_shap/plugins/__init__.py]]"
  - "[[src/nlp_shap/plugins/backends.py]]"
  - "[[src/nlp_shap/plugins/builtin.py]]"
  - "[[src/nlp_shap/plugins/context.py]]"
  - "[[src/nlp_shap/plugins/groups.py]]"
  - "[[src/nlp_shap/plugins/options.py]]"
  - "[[src/nlp_shap/plugins/registry.py]]"
---

# plugins

Part of [[nlp-shap]]. Registered plugins: [[Plugins]].

## Responsibility

- Map `(group, name)` to a factory, from the built-in table and from packaging entry points, and build plugins with a `PluginContext` that carries their validated options ([[Plugin options]]).

## Public surface

- `PluginGroup`: eight entry-point groups (`nlp_shap.estimators`, `estimands`, `value_fns`, `normalizers`, `backends`, `partitions`, `absence_policies`, `renderers`).
- `PluginContext(config, options, registry)`: `option(key, default)` returns the option or the default, accepts an integer for a float default, raises `ConfigError` on another type. Plugin classes take it as their first constructor argument.
- `PluginFactory = Callable[[PluginContext], object]`.
- `PluginRegistry`: `register(group, name, factory, replace=False)` (raises `PluginConflictError` for a duplicate), `resolve(group, name, context)` (selects and validates the plugin's options, then calls the factory with a context narrowed to them), `resolve_factory` (`PluginNotFoundError`), `resolve_type`, `names(group)`, `load_entry_points(group)` (skips names already registered without importing them; wraps import failures in `PluginLoadError`).
- `BUILTIN_PLUGINS`: the built-in factories of all eight groups, all four backends included; mirrors `pyproject.toml`. `register_builtin_plugins(registry)`; `default_registry()` = builtins plus third-party entry points, used by `ExplainRunner`, `viz.render` and the CLI.
- `plugins/options.py`: `group_key(group)`, `select_plugin_options` (`<group>/<name>` first, then the bare name; both present raises), `validate_plugin_options` (against `options_model` of a `DeclaresPluginOptions` factory), `run_plugins(config)`, `check_plugin_options(config, registry)` (every key must name a plugin of the run or a registered renderer, a bare name must be unambiguous, values must validate).
- `instantiate_backend(context)`: resolves the backend named by `context.config.backend.kind`.

## Data flow

- In: plugin names and `explanation.plugin_options` from the [[pipeline]] configuration.
- Out: plugin instances; call sites cast `object` to the [[protocols]] type.
- `plugins/context.py` imports `ExplainConfig` only under `TYPE_CHECKING`, so there is no `plugins` and `pipeline` import cycle.

## Invariants

- Built-in names resolve with or without entry points; entry points never shadow a built-in name.
- Unknown option keys, wrong types and keys naming no plugin of the run fail with `ConfigError` before any generation.

## Known issues

- None open.

## Tests

- [[tests/plugins/test_registry.py]]
- [[tests/plugins/test_builtin.py]]
- [[tests/plugins/test_groups.py]]
- [[tests/plugins/test_backends.py]]
