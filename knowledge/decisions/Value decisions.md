---
title: "Value decisions"
type: decision-log
domain: value
created: 2026-10-08
updated: 2026-10-08
tags: [decisions, value]
related-code:
  - "[[src/mllm_shap/value/generation_records.py]]"
  - "[[src/mllm_shap/value/logprob.py]]"
  - "[[src/mllm_shap/backends/mock/backend.py]]"
  - "[[src/mllm_shap/pipeline/orchestrator.py]]"
  - "[[src/mllm_shap/pipeline/reanalyze.py]]"
---

# Value decisions

Part of [[mllm-shap]]; architecture: [[value]]. Related: [[backends]], [[pipeline]], [[Reanalyze]].

## 2026-10-08 — Logprob value is the coalition's own mean token logprob

Decision: `logprob` scores `v(S)` as the mean natural-log probability of the tokens generated for coalition `S`, as returned by the backend; an empty generation scores 0 and the base generation is not read. It requires `SupportsLogprobs` (`ConfigError` before generating), raises `LogprobsUnavailableError` on a record without logprobs, and archives store them per row. The hash fallback is deleted.

Why: `score(base, candidate)` already read the candidate's own logprobs and ignored the base, so the design scores the coalition's generation. Scoring the base answer under each coalition prompt needs forced decoding, which no backend protocol offers and chat APIs do not serve. A mean, not a sum, keeps confidence apart from generation length.

Consequences: Attributions measure confidence, not agreement with the base answer. Rejected: keeping the hash stub, which attributed noise.

## 2026-10-08 — Contextual hash embedding derived from text only

Decision: `contextual_embedding = hash_embedding(text + "\0")` and `embedding = hash_embedding(text)`, so every field of a text-derived record depends on the generated text only. `context_id` stays in the signature, is ignored, emits a `DeprecationWarning` and is removed in 0.2.0. The backends, the orchestrator and `reanalyze` call `generation_record_from_text(text)`. `_with_embedding_source` is unchanged: backend vectors equal to the text-derived ones count as `hash`, model vectors (transformers) as `backend`, which `reanalyze --strict` still refuses.

Why: `generation_record_from_text(text, context_id)` salted `contextual_embedding` with `context_id`. The mock, LM Studio and application programming interface (API) backends passed the rendered coalition snapshot's id; the orchestrator and `reanalyze` passed the explained snapshot's id. So a contextual run scored vectors `reanalyze` cannot rebuild and was recorded as `embedding_source: backend` ([[Fixed issues|C15]]).

Consequences: Mock, LM Studio and API runs give identical `explain` and `reanalyze` values in both embedding modes and record `embedding_source: hash` (test `test_reanalyze_matches_explain_embedding_values` in [[tests/pipeline/test_runner_e2e.py]]). Their contextual-mode values change; static-mode values do not. On the four-token C15 repro (`mllm-shap explain` and `mllm-shap reanalyze`, run on 2026-10-08) both give `[-0.227, 0.243, 0.474, 0.494]` and `--strict` exits 0. In hash mode the contextual vector no longer differs between snapshots producing the same text and differs from the static one only by the separator. Third-party backends passing `context_id` keep working with a warning. Rejected: salting with the explained snapshot id in the backends (the backend protocol never sees it; adding it changes every backend's signature); rebuilding contextual vectors in `reanalyze` (archives store the snapshot id, not its text); documenting the mismatch (`--strict` would refuse every contextual run of the continuous integration backend).
