---
title: "Backends decisions"
type: decision-log
domain: backends
created: 2026-10-07
updated: 2026-10-09
tags: [decisions, backends]
related-code:
  - "[[src/nlp_shap/protocols/value.py]]"
  - "[[src/nlp_shap/pipeline/orchestrator.py]]"
  - "[[src/nlp_shap/backends/transformers/text.py]]"
---

# Backends decisions

Part of [[nlp-shap]]; architecture: [[backends]]. Related: [[value]], [[Capability protocol]].

## 2026-10-09 — Half-precision prefix cache warns instead of promising equality

Decision: `TransformersTextBackend` keeps the prefix cache in float16 and bfloat16 and logs one `WARNING` per backend that cached results can differ from `kv_cache` off; float32 is the documented setting for identical results.

Why: The cached path splits the prompt over several forward passes and the uncached path uses one; half-precision attention rounds them differently and can flip a near-tied greedy token (D04: up to 0.0138 on Qwen2.5-3B float16 on Apple Metal Performance Shaders (MPS)). The guide claimed equality.

Consequences: Default `dtype: auto` runs on accelerators keep the speed-up with a visible caveat; `test_transformers_backend_warns_once_on_half_precision_prefix_cache` pins it ([[Fixed issues]]). Rejected: disabling the cache in half precision (loses the speed-up on every accelerator); a float32 prefix pass (different kernels, still not bit-identical).

## 2026-10-07 — Embeddings computed only when the value function needs them

Decision: Value functions that score embeddings declare it through the `RequiresEmbeddings` capability (`needs_embeddings`). The orchestrator then sets `GenerationConfig.compute_embeddings`, backends compute embeddings only when it is set, and the orchestrator swaps the backend's embeddings into the text-derived scoring record.

Why: The transformers backend ran two extra forward passes per coalition (static and contextual embeddings) under the model lock, and the pipeline then scored hash embeddings of the text, discarding the work.

Consequences: Text-based runs (the default `tfidf_cosine`) skip both passes. A transformers run's embedding values differ between `explain` (model embeddings) and `reanalyze` (hash embeddings of archived text), because archives store text only; the 0.1.18 release notes say so. Rejected: always computing embeddings, which pays two passes per coalition for value functions that never read them.
