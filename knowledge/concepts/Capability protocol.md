---
title: "Capability protocol"
type: concept
domain: protocols
tags: [concept]
created: 2026-10-08
related-code:
  - "[[src/mllm_shap/protocols/backend.py]]"
  - "[[src/mllm_shap/protocols/value.py]]"
---

# Capability protocol

A capability protocol is an optional, runtime-checkable `typing.Protocol` a plugin may satisfy on top of its required protocol; the pipeline tests it with `isinstance` instead of probing attributes. The capabilities are `SupportsKvCache`, `SupportsLogprobs` and `SupportsAclose` for backends and `FittableValueFunction`, `RequiresEmbeddings` and `RequiresLogprobs` for value functions; a value function that needs logprobs on a backend without `SupportsLogprobs` raises `ConfigError` before any generation.

Package: [[protocols]]. Decision: [[Backends decisions#2026-10-07 — Embeddings computed only when the value function needs them|Embeddings computed only when the value function needs them]].
