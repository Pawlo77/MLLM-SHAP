---
title: "D05 - Prefix cache speed-up fell below its threshold"
type: issue
domain: backends
status: open
severity: minor
found: 2026-10-10
created: 2026-10-10
fixed_in: ""
tags: [issue, kv-cache, performance]
related-code:
  - "[[src/nlp_shap/backends/transformers/text.py]]"
  - "[[tests/practical/deep/test_runtime.py]]"
---

# D05 - Prefix cache speed-up fell below its threshold

Package: [[backends]].

## Symptom

- The deep practical test `deep.runtime.kv_cache` (Qwen2.5-3B float16, Apple Metal Performance Shaders (MPS)) asserts that a run with `kv_cache.enabled` is at least 1.1 times faster than without it.
- Measured speed-ups (seconds cache on and off): 1.17 (47.5, 55.6) and 1.13 (75.5, 85.3) on 2026-10-09; 0.93 (89.7, 83.6), 1.02 (88.3, 89.8) and 1.02 (91.1, 93.3) on 2026-10-10, on `main` and on the alignment branch alike.
- Cached and uncached values still agree within the asserted gap (0.0138, limit 0.02).

## Cause

- Unknown. Both arms slowed by about 1.7 times while the code path did not change, which points at the machine (thermal state, accelerator memory) more than at `TransformersTextBackend._attach_prefix_cache`; the cached arm lost more, consistent with the cost of `copy.deepcopy` of key/value tensors growing under load.

## Repro

```bash
make practical TIER=deep FEATURE=deep.runtime.kv_cache
```

## Fix

- Open. Re-measure on an idle, cool machine; if the speed-up stays below 1.1, profile the deep copy of cached states (in-place cropping or copying only the cropped prefix).
