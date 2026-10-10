---
title: "backends"
type: module
domain: backends
tags: [architecture, backends]
created: 2026-10-07
updated: 2026-10-09
related-code:
  - "[[src/nlp_shap/backends/__init__.py]]"
  - "[[src/nlp_shap/backends/_chat.py]]"
  - "[[src/nlp_shap/backends/api/__init__.py]]"
  - "[[src/nlp_shap/backends/api/backend.py]]"
  - "[[src/nlp_shap/backends/api/payload.py]]"
  - "[[src/nlp_shap/backends/lmstudio/__init__.py]]"
  - "[[src/nlp_shap/backends/lmstudio/backend.py]]"
  - "[[src/nlp_shap/backends/lmstudio/chat.py]]"
  - "[[src/nlp_shap/backends/lmstudio/models.py]]"
  - "[[src/nlp_shap/backends/lmstudio/options.py]]"
  - "[[src/nlp_shap/backends/mock/__init__.py]]"
  - "[[src/nlp_shap/backends/mock/backend.py]]"
  - "[[src/nlp_shap/backends/transformers/__init__.py]]"
  - "[[src/nlp_shap/backends/transformers/_torch.py]]"
  - "[[src/nlp_shap/backends/transformers/chat.py]]"
  - "[[src/nlp_shap/backends/transformers/embeddings.py]]"
  - "[[src/nlp_shap/backends/transformers/options.py]]"
  - "[[src/nlp_shap/backends/transformers/text.py]]"
---

# backends

Part of [[nlp-shap]].

## Responsibility

- Generate assistant text for a rendered snapshot and return it as a `GenerationRecord`; optionally reuse prompt-prefix state and compute model embeddings.

## Public surface

- All backends take a `PluginContext` and read `context.config.backend`; `generate(snapshot, generation: GenerationConfig)`.
- `MockBackend` (`mock`): deterministic text from a SHA-256 of the snapshot id, each message's `message_chat_text`, model id and generation parameters; at most 16 four-character tokens; used in continuous integration (CI).
- `ApiBackend` (`api`): OpenAI-compatible application programming interface (API) `/chat/completions` over `httpx`, with `Timeout(backend.timeout_s)` and at most `max_inflight` connections.
  - Retries `RETRYABLE_STATUS` (408, 429, 500, 502, 503, 504) and transport errors up to `backend.max_retries` with exponential backoff, honouring `Retry-After` up to `timeout_s`. Non-JSON responses raise `BackendUnavailableError`.
  - Sends `"logprobs": true`; `extract_logprobs` reads `choices[0].logprobs.content[*].logprob`, else `None`. `returns_logprobs` is true (`SupportsLogprobs`).
  - Caches temperature-0 records (text and logprobs) in an LRU of `RESPONSE_CACHE_SIZE` (1024).
  - `backend.api_host` is required (`ConfigError` without it; no environment or localhost fallback).
  - Bearer key `NLP_SHAP_API_KEY` for any host; `OPENAI_API_KEY` only for `OPENAI_API_HOST` (`api.openai.com`), so an OpenAI key never reaches another server.
- `LmStudioBackend` (`lmstudio`): local LM Studio server through the `lmstudio` SDK.
  - `LmStudioOptions` (`hub_search`, `quantization`, `auto_download`, `serialize_generate`) come from `backend.options` (`parse_lmstudio_options`; unknown keys raise `ConfigError`).
  - `match_downloaded_model_key` matches exactly, then by the prefix of the last path segment, then by substring, and raises on ambiguity.
  - `pick_search_result` raises `BackendUnavailableError` (listing the hits) when no hub hit is exact or matches the name, so `auto_download` never fetches an unrelated model.
  - Retries once, only on transient connection errors (`_is_transient`).
  - Records carry no logprobs: the SDK 1.5 `PredictionResult` exposes none.
- `TransformersTextBackend` (`transformers`): Hugging Face causal language model on CUDA (Compute Unified Device Architecture), Apple MPS (Metal Performance Shaders) or CPU.
  - `TransformersOptions`: `dtype` (`auto|float32|float16|bfloat16`; `auto` picks bfloat16 where supported, float16 on other accelerators, float32 on CPU) and `revision`, from `backend.options`.
  - Tokenisation, generation and decoding run under one model lock.
  - With the key-value (KV) cache enabled (`SupportsKvCache`), `_attach_prefix_cache` copies and crops the longest cached prefix and runs only the uncovered tokens. It stores the result and hands the state to `generate`; `kv_cache_hits` and `kv_cache_reused_tokens` count the reuse.
  - Static and contextual embeddings are computed only when `generation.compute_embeddings` is set.
  - `generate` returns raw logits (`output_logits`); `_token_logprobs` takes their log-softmax at each generated token, end-of-sequence included. `returns_logprobs` is true.
- `backends/_chat.py`: `snapshot_has_audio` (checks for `AudioPart`), `message_chat_text` (`"<speaker>: <text>"` when a speaker is set and the text is not empty; [[Domain decisions#2026-10-08 — Speaker rendered into the message content|speaker rendering]]), `snapshot_to_chat_messages`; `lmstudio/chat.py` `snapshot_to_chat`; `transformers/chat.py` `render_prompt`; `transformers/_torch.py` `import_torch`.

## Data flow

- In: rendered `ConversationSnapshot` and `GenerationConfig` from the [[pipeline]] orchestrator; `PluginContext` at construction.
- Out: `GenerationRecord` built with `value.generation_records.generation_record_from_text(text)` ([[value]]), with model embeddings when requested; the mock, LM Studio and API backends have none, so their records equal the ones `reanalyze` builds from the archived text.
- Optional dependencies (`httpx`, `lmstudio`, `torch`, `transformers`) are imported lazily; a missing extra raises `BackendUnavailableError` ([[core]]).

## Invariants

- The API, LM Studio and transformers backends refuse audio parts with `ValueError`; the mock backend hashes text only ([[D02 - Audio backend]]).
- In float32, greedy transformers outputs are identical with the prefix cache on and off, and their logprobs agree within 1e-4; in float16 and bfloat16 they agree up to rounding, and the backend warns once ([[Backends decisions#2026-10-09 — Half-precision prefix cache warns instead of promising equality|decision]]). No prompt token is processed twice.
- A backend injected into `ExplainRunner` is reused across calls and closed by the runner; otherwise one backend is built and closed per `explain()`.

## Known issues

- None open. Fixed: C06, C08, C15, D04 ([[Fixed issues]]).

## Tests

- [[tests/backends/test_mock.py]]
- [[tests/backends/test_api.py]]
- [[tests/backends/test_lmstudio.py]]
- [[tests/backends/test_lmstudio_chat.py]]
- [[tests/backends/test_lmstudio_models.py]]
- [[tests/backends/test_lmstudio_options.py]]
- [[tests/backends/test_lmstudio_resolve.py]]
- [[tests/backends/test_transformers_chat.py]]
- [[tests/backends/test_transformers_text.py]]
- [[tests/backends/test_transformers_gpu.py]]
- [[tests/benchmarks/test_api_backend_bench.py]]
