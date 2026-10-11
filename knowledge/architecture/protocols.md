---
title: "protocols"
type: module
domain: protocols
parent: "[[mllm-shap]]"
tags: [architecture, protocols]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/protocols/__init__.py]]"
  - "[[src/mllm_shap/protocols/backend.py]]"
  - "[[src/mllm_shap/protocols/embedding.py]]"
  - "[[src/mllm_shap/protocols/estimand.py]]"
  - "[[src/mllm_shap/protocols/estimator.py]]"
  - "[[src/mllm_shap/protocols/masking.py]]"
  - "[[src/mllm_shap/protocols/normalizer.py]]"
  - "[[src/mllm_shap/protocols/partition.py]]"
  - "[[src/mllm_shap/protocols/plugin.py]]"
  - "[[src/mllm_shap/protocols/renderer.py]]"
  - "[[src/mllm_shap/protocols/value.py]]"
---

# protocols

## Responsibility

- `typing.Protocol` contracts each plugin kind satisfies structurally, plus runtime-checkable capability protocols the pipeline tests with `isinstance` instead of probing attributes ([[Capability protocol]]).

## Public surface

- `GenerativeBackend`: `model_id`, `async generate(snapshot, generation: GenerationConfig) -> GenerationResult`; `GenerationResult` promises `text`.
- Capabilities: `SupportsKvCache` (`kv_cache_hits`, `kv_cache_reused_tokens`, `set_kv_cache_enabled(enabled, max_entries)`), `SupportsLogprobs` (`returns_logprobs`; the `transformers` and `api` backends), `SupportsAclose` (`async aclose()`), `FittableValueFunction` (`fit(corpus)`), `RequiresEmbeddings` (`needs_embeddings`), `RequiresLogprobs` (`needs_logprobs`).
- `EstimatorStrategy`: `name`, `bind_snapshot`, `sample_masks(player_set, budget_fraction, include_minimal_masks, seed)`, `estimate_attributions(masks, payoffs, aggregator)` ([[estimation|Estimator]]); `TwoPhaseEstimator`: `begin_allocation(masks, payoffs)`, `sample_allocation_masks()`.
- `EstimandAggregator`: `estimand`, `coalition_weight(size, n)`, `aggregate(masks, payoffs)` ([[estimation|Estimand]]).
- `ValueFunction`: `name`, `score(base, candidate)` ([[value|Value function]]).
- `AbsencePolicy`: `name`, `apply(snapshot, players, mask)` ([[masking|Absence policy]]); `PlayerPartitioner`: `name`, `partition(snapshot) -> PlayerSet` with spans ([[masking|Partition]]).
- `Normalizer`, `AttributionRenderer`, `EmbeddingProvider`.
- `protocols/plugin.py`: `PluginOptions` (pydantic model with `extra="forbid"`, strict types, frozen; the base class accepts no keys) and `DeclaresPluginOptions` (a plugin class with an `options_model` class variable) ([[Plugin options]]).

## Data flow

- Imports [[domain]] types only; the [[pipeline]] and [[plugins]] packages type against these protocols and cast registry results to them.

## Invariants

- A plugin that satisfies the protocols works without subclassing; capabilities are optional and detected with `isinstance`.
- The orchestrator has no estimator-name switch: every estimator exposes `estimate_attributions`, and two-phase estimators are found by `TwoPhaseEstimator`.

## Known issues

- None open.

## Tests

- [[tests/pipeline/test_capabilities.py]]
- [[tests/plugins/test_registry.py]]
- [[tests/pipeline/test_runner_e2e.py]]
