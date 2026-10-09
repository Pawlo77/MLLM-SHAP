---
title: "Prefix hash"
type: concept
domain: runtime
tags: [concept]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/nlp_shap/runtime/kv_cache.py]]"
---

# Prefix hash

The prefix hash of a coalition is the first 16 hex characters of a SHA-256 over the snapshot id and the presence of every player except the last. Coalitions that differ only in the last player render the same prompt up to its span, so with the key-value (KV) cache enabled the orchestrator schedules jobs with the same prefix hash next to each other and a backend that reuses prompt prefixes can hit its cache.

Decision: [[Runtime decisions#2026-10-07 — KV cache off by default|KV cache off by default]]; fixed issues: C06, C11 ([[Fixed issues]]).
