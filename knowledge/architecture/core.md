---
title: "Package root"
type: module
domain: core
tags: [architecture, core]
created: 2026-10-07
updated: 2026-10-10
related-code:
  - "[[src/mllm_shap/__init__.py]]"
  - "[[src/mllm_shap/_logging.py]]"
  - "[[src/mllm_shap/cli.py]]"
  - "[[src/mllm_shap/errors.py]]"
---

# Package root

Part of [[mllm-shap]].

## Responsibility

- **Public surface**: `mllm_shap/__init__.py` re-exports what a user needs for a run; the full list is [[Public API]].
- **Command line**: `cli.py` is the `mllm-shap` console script (`[project.scripts]`).
- **Logging**: `_logging.py` keeps the package quiet on import and lets users opt in.
- **Errors**: `errors.py` holds the typed error hierarchy.

## Public surface

- `__all__`: 71 names, listed in [[Public API]]: domain types, every built-in plugin class, the pipeline entry points, plugin machinery, every error, `ObservabilitySink`, `InMemoryObservabilitySink`, `CoalitionProgress`, `configure_logging` and `__version__`.
- `render_attribution`, `render_attribution_html`, `display_attribution_html`: resolved lazily through module `__getattr__` (listed by `__dir__`), so `import mllm_shap` works without the `viz` extra.
- `cli.main(argv)`: subcommands `explain` (JSON result, optional PNG chart), `reanalyze [--strict]`, `plugins` and `config --example | --check`. `--strict` raises `ReanalyzeMismatchError` for runs scored with backend embeddings instead of logging a warning.
  - Exit codes: `EXIT_OK` 0, `EXIT_INVALID` 2 (arguments, `--log-level` outside `LOG_LEVELS`, configuration, snapshot, archive), `EXIT_BACKEND_UNAVAILABLE` 3 (`BackendUnavailableError`, `ConnectionError`); other exceptions propagate.
  - `EXAMPLE_CONFIG` is the commented YAML printed by `config --example`.
- `bootstrap_logging()`: runs once at import; applies `[tool.logging]` from the file named by `MLLM_SHAP_LOGGING_CONFIG`, otherwise installs only a `NullHandler`.
- `configure_logging(config_path=None, level=None, json=False)`: TOML configuration, or with `level`/`json` a single stderr handler (`JsonLogFormatter` for one JSON object per line) and `propagate=False`; `json` with `config_path` raises `ValueError`.
- Errors (13 classes), each also subclassing the matching builtin (`LookupError`, `ValueError`, `RuntimeError`, `FileExistsError`, `FileNotFoundError`, `ImportError`):
  - `MllmShapError` root; `PluginNotFoundError`, `PluginConflictError`, `PluginLoadError` (`entry_point_name`), `ConfigError`, `BackendUnavailableError` and its subclass `LogprobsUnavailableError` (a record without the logprobs `logprob` needs; command-line exit code 3).
  - `ArchiveExistsError`, `ArchiveIncompleteError`, `ArchiveNotFoundError` (no manifest or SQLite database), `ArchiveCorruptError` (unreadable archive or manifest), `ManifestVersionError`, `ReanalyzeMismatchError`.

## Data flow

- Importing `mllm_shap` imports [[domain]], [[estimation]], [[masking]], [[pipeline]], [[plugins]], [[runtime]], [[value]] and the backend classes (their heavy dependencies are imported lazily), then calls `bootstrap_logging()`.
- `cli.main` reads YAML (`explain_config_from_yaml`) and snapshot JSON (`ConversationSnapshot.from_json`), runs `ExplainRunner.explain_sync` or `reanalyze`, and writes `ExplainRunOutput.to_dict()` as JSON; `--png` renders through [[viz]].
- `__version__` comes from installed package metadata (`0.0.0+unknown` when not installed).

## Invariants

- Exported names import without optional extras and without touching the host's logging configuration.
- Every error raised on purpose is an `MllmShapError` and still matches the builtin exception callers caught before 0.1.18.
- The CLI never prints a traceback for invalid input or an unavailable backend.

## Known issues

- None open. The logging-at-import problem from the audit was fixed in 0.1.18.

## Tests

- [[tests/test_smoke.py]]
- [[tests/test_public_api.py]]
- [[tests/test_cli.py]]
- [[tests/test_logging.py]]
- [[tests/test_base_install.py]]
- [[tests/domain/test_imports.py]]
- [[tests/docs/test_public_exports.py]]
