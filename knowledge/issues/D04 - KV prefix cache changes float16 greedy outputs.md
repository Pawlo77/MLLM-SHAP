---
title: "D04 - KV prefix cache changes float16 greedy outputs"
type: issue
domain: backends
status: open
severity: minor
found: 2026-10-08
created: 2026-10-08
fixed_in: ""
tags: [issue, backends, kv-cache]
related-code:
  - "[[src/nlp_shap/backends/transformers/text.py]]"
  - "[[src/nlp_shap/runtime/kv_cache.py]]"
---

# D04 - KV prefix cache changes float16 greedy outputs

Package: [[backends]].

## Symptom

- `HuggingFaceTB/SmolLM2-360M-Instruct` in float16 on Metal Performance Shaders (MPS): an exact `players: messages` run over three turns (64 coalitions) gives values up to 9.9e-4 apart with `kv_cache.enabled` true and false, so some greedy generations differ.
- The same run in float32, and `Qwen/Qwen2.5-1.5B-Instruct` in float16, agree exactly.
- `Qwen/Qwen2.5-3B-Instruct` in float16 on MPS: exact over six message players of a 10-turn thread with 64-token answers gives values up to 0.0138 apart.

## Cause

- `TransformersTextBackend._attach_prefix_cache` runs the cached prefix and the remaining prompt tokens in separate forward passes. In float16 those passes round attention sums differently from one pass, which can flip a near-tied greedy token.

## Repro

- Pinned by the practical test `backend.transformers.kv_cache`: it asserts float32 equality and records the float16 gap as `float16_max_gap` in `tests/practical/results/latest.json`.
- `deep.runtime.kv_cache` records the 3B gap as `max_gap` and asserts it at most 0.02 (float32 3B weights exceed the deep tier's 8 GB cap).
- No unit test: tiny random weights in float32 do not show it.

## Fix

- Open. Options: document float32 for exact cache equivalence, or run the uncached prefix in one pass in half precision.
