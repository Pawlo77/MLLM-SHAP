---
title: "pipeline"
type: module
domain: pipeline
parent: "[[mllm-shap]]"
tags: [architecture, pipeline]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/pipeline/__init__.py]]"
  - "[[src/mllm_shap/pipeline/config.py]]"
  - "[[src/mllm_shap/pipeline/context.py]]"
  - "[[src/mllm_shap/pipeline/manifest.py]]"
  - "[[src/mllm_shap/pipeline/orchestrator.py]]"
  - "[[src/mllm_shap/pipeline/reanalyze.py]]"
  - "[[src/mllm_shap/pipeline/result.py]]"
  - "[[src/mllm_shap/pipeline/runner.py]]"
---

# pipeline

Walkthrough: [[How a run works]].

## Responsibility

- User entry point (`ExplainRunner`), configuration schema, the orchestration of one explain run, re-scoring of an archive, and the result and manifest types.

## Public surface

Signatures: [[Public API]]; configuration keys: [[Config keys]].

- `ExplainConfig`: `backend`, `generation` and `explanation` sections. `BudgetConfig` takes `fraction` or `evaluations`, never both; only the key that is set is serialised, so a fraction-only configuration keeps its [[Run id]] ([[Pipeline decisions#2026-10-08 — Absolute budget passed to samplers as an exact fraction|Absolute budget passed to samplers as an exact fraction]]).
- `ExplanationConfig` cross-field rules raise `ConfigError`: `exact` needs `budget.fraction: 1.0`, `complementary` and `neyman_cc` need the Shapley estimand, `neyman_cc` refuses `include_minimal_masks`. `explain_config_from_yaml` raises `ConfigError` for an invalid, empty or non-mapping document.
- `ExplainRunner`: `explain`, `explain_sync` (a `RuntimeError` inside a running loop), `reanalyze`; `close`, `aclose` and both context managers close an injected backend and the runner's own event loop.
- `ExplainOrchestrator.run()`: the data flow below. `score_coalitions` fits a `FittableValueFunction` on base plus coalitions and scores each coalition; `reanalyze` uses it too.
- `reanalyze_sync` ([[Reanalyze]]): for a run scored with backend embeddings and a `RequiresEmbeddings` value function it logs a `WARNING`, or raises `ReanalyzeMismatchError` with `strict`; a `RequiresLogprobs` value function on rows without logprobs raises `ReanalyzeMismatchError`.
- `ExplainResult`; `ExplainRunOutput` with `to_dict()` (`ExplainRunPayload`; each label adds `speaker` and `label`, the span's `display_label`); `player_set` is `None` after `reanalyze`.
- `RunManifest`, `parse_manifest`, `MANIFEST_SCHEMA_VERSION` 4 ([[Manifest]]): `embedding_source` is `"none"`, `"hash"` or `"backend"`; wrong field types raise `ArchiveCorruptError`.
- `ExplainContext`: resolved inputs of one run; `plugin_context()` builds the `PluginContext`.

## Data flow

1. `ExplainRunner._build_context` checks plugin options, resolves the partitioner, refuses a plan above `2 ** max_players` coalitions and sets the [[Run id]] from the configuration without its archive section plus the model id.
2. Without an injected backend the runner builds one through [[plugins]] `instantiate_backend` and closes it after the call.
3. The orchestrator resolves the plugins. A `RequiresLogprobs` value function on a backend without `SupportsLogprobs` raises `ConfigError` ([[Value decisions#2026-10-08 — Logprob value is the coalition's own mean token logprob|logprob value]]). A `RequiresEmbeddings` value function switches `generation.compute_embeddings` on; the archive opens with the manifest.
4. It generates the [[Base generation]] and writes it to the archive first.
5. It samples masks with `budget_fraction(n, fraction, evaluations)` from [[estimation]] and adds a missing empty or grand coalition; the base text is `v(N)`. A `SupportsKvCache` backend gets its prefix cache enabled, and the metrics count only this run's hits.
6. It plans `(mask, coalition key, prefix hash)` per mask, sorted by prefix hash only with `kv_cache.enabled`; the [[runtime]] scheduler consumes lazily rendered jobs, and records come back by coalition key in mask order.
7. `score_coalitions` scores records built with `generation_record_from_text(text)` plus the backend's logprobs, kept per coalition key in the store. Hash vectors depend on the text only, so the records equal what `reanalyze` builds from the archived text and logprobs.
8. A `TwoPhaseEstimator` gets `begin_allocation(masks, payoffs)`; phase-two masks run the same way and every coalition is rescored on the joint corpus.
9. It records `embedding_source` (`"backend"` when a backend vector differs from the hash vector, else `"hash"` when embeddings are scored), marks the archive complete, aggregates with `estimator.estimate_attributions`, normalises and logs each stage at `INFO`.

## Invariants

- The empty and grand coalitions are always evaluated; `sum(values) == grand_value - base_value` for Shapley on the full table.
- Payoffs are aligned with masks whatever order the scheduler ran the jobs in ([[Fixed issues|C11]]).
- The base generation is in the archive before any coalition row; an archive is marked complete only after every coalition of both phases is stored.
- `reanalyze` refuses a different estimator, absence policy or partition (`ReanalyzeMismatchError`). It raises `ArchiveIncompleteError` for a foreign or unfinished archive, a missing boundary coalition, or missing rows under a full-table budget.
- `explain` and `reanalyze` score identical records for a run without model embeddings ([[Fixed issues|C15]]).

## Known issues

- Fixed: C01, C02, C04, C08, C11, C15, D01 ([[Fixed issues]]).

## Tests

- [[tests/pipeline/test_runner_e2e.py]]
- [[tests/pipeline/test_runner_backend.py]]
- [[tests/pipeline/test_runner_archive_integrity.py]]
- [[tests/pipeline/test_runner_progress.py]]
- [[tests/pipeline/test_orchestrator.py]]
- [[tests/pipeline/test_pipeline_axioms.py]]
- [[tests/pipeline/test_reanalyze.py]]
- [[tests/pipeline/test_capabilities.py]]
- [[tests/pipeline/test_config.py]]
- [[tests/pipeline/test_budget_evaluations.py]]
- [[tests/pipeline/test_dev_configs.py]]
- [[tests/pipeline/test_manifest.py]]
- [[tests/pipeline/test_result.py]]
- [[tests/test_text_mvp_checklist.py]]
- [[tests/benchmarks/test_pipeline_bench.py]]
