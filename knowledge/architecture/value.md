---
title: "value"
type: module
domain: value
tags: [architecture, value]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/value/__init__.py]]"
  - "[[src/mllm_shap/value/_metrics.py]]"
  - "[[src/mllm_shap/value/_token_hash.py]]"
  - "[[src/mllm_shap/value/embedding.py]]"
  - "[[src/mllm_shap/value/generation_records.py]]"
  - "[[src/mllm_shap/value/logprob.py]]"
  - "[[src/mllm_shap/value/tfidf.py]]"
---

# value

Part of [[mllm-shap]].

## Responsibility

- Score one coalition generation against the [[Base generation]]; the score is the coalition payoff `v(S)` (Value function).
- Build the generation records both `explain` and `reanalyze` score.

## Public surface

- `TfIdfCosineValue` (`tfidf_cosine`, the default): a `FittableValueFunction`. `fit(corpus)` freezes inverse document frequencies over the base and all coalition records. `score` is the cosine similarity of term frequency-inverse document frequency (TF-IDF) vectors built from `text_token_rows` of `GenerationRecord` inputs.
- `EmbeddingValue`: abstract base holding the embedding mode (`explanation.embedding_mode` through the plugin context), the provider and `score`; `needs_embeddings` is true, so subclasses satisfy `RequiresEmbeddings` ([[Backends decisions#2026-10-07 — Embeddings computed only when the value function needs them|Embeddings computed only when the value function needs them]]). `CosineEmbeddingValue` (`embedding_cosine`) and `EuclideanEmbeddingValue` (`embedding_euclidean`, `1 / (1 + distance)`) set `registered_name` and the metric. Both read `embedding`, or `contextual_embedding` in `contextual` mode.
- `LogprobValue` (`logprob`): mean of the candidate's `logprobs` (0 for an empty generation); ignores the base; `needs_logprobs` makes it `RequiresLogprobs`; a record with `logprobs is None` raises `LogprobsUnavailableError` ([[Value decisions#2026-10-08 — Logprob value is the coalition's own mean token logprob|logprob value]]).
- `generation_record_from_text(text, context_id=None)`: token rows from `text.split()` with `stable_token_id` (56-bit `blake2b`) and 8-dimensional `hash_embedding` vectors. `embedding = hash_embedding(text)` and `contextual_embedding = hash_embedding(text + "\0")`, so every field depends on the generated text only. `context_id` is ignored and emits a `DeprecationWarning` (removal in 0.2.0; [[Value decisions#2026-10-08 — Contextual hash embedding derived from text only|Contextual hash embedding derived from text only]]).
- Private helpers: `_metrics.py` (cosine and Euclidean similarity), `_token_hash.py` (`TokenHasher`, maps token rows to dense ids).

## Data flow

- In: records from `generation_record_from_text`, built by the [[pipeline]] orchestrator from backend text (with backend logprobs, and model embeddings when requested) and by `reanalyze` from archived text and logprobs.
- Out: one float per coalition.
- Imports [[domain]], [[protocols]], [[plugins]] (context type only).

## Invariants

- `explain` and `reanalyze` score the same records for text-based value functions and `logprob`, and for embedding value functions when the backend has no model embeddings. They return the same values on the same archive ([[Fixed issues|C15]]).
- TF-IDF weights are fitted once per scoring pass on exactly the corpus that is scored (both Neyman phases are rescored together).
- `score(base, base)` is the grand-coalition payoff; for `tfidf_cosine` it is 1 for any non-empty base text.

## Known issues

- Embedding values of a transformers run differ between `explain` (model embeddings) and `reanalyze` (hash embeddings of archived text), because archives store text only; documented in the 0.1.18 release notes.
- Fixed: C08, C15 ([[Fixed issues]]).

## Tests

- [[tests/value/test_tfidf.py]]
- [[tests/value/test_embedding.py]]
- [[tests/value/test_logprob.py]]
- [[tests/value/test_metrics.py]]
- [[tests/value/test_generation_records.py]]
- [[tests/benchmarks/test_value_bench.py]]
