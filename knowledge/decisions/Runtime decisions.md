---
title: "Runtime decisions"
type: decision-log
domain: runtime
parent: "[[mllm-shap]]"
created: 2026-10-07
updated: 2026-10-08
tags: [decisions, runtime]
related-code:
  - "[[src/mllm_shap/pipeline/config.py]]"
  - "[[src/mllm_shap/runtime/kv_cache.py]]"
---
# Runtime decisions

Architecture: [[runtime]]. Concept: [[Prefix hash]].

## 2026-10-07 — KV cache off by default

Decision: `KvCacheConfig.enabled` defaults to `False` with a `max_entries` bound of 8, shipped in 0.1.18 together with real prefix reuse in the transformers backend.

Why: The transformers key-value (KV) cache was on by default but never reused, and computing it was quadratic in prompt length ([[Fixed issues|C06]]).

Consequences: Default transformers runs got faster at once; users opt in to reuse. Job grouping by prefix hash happens only when the cache is enabled. Rejected: keeping it on while fixing reuse, which leaves the default slow on every release until the fix is benchmarked.
