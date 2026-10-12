---
title: "backends"
type: module
domain: backends
parent: "[[mllm-shap]]"
tags: [architecture, backends]
created: 2026-10-07
updated: 2026-10-12
related-code:
  - "[[src/mllm_shap/backends/__init__.py]]"
  - "[[src/mllm_shap/backends/_chat.py]]"
  - "[[src/mllm_shap/backends/api/__init__.py]]"
  - "[[src/mllm_shap/backends/api/backend.py]]"
  - "[[src/mllm_shap/backends/api/payload.py]]"
  - "[[src/mllm_shap/backends/lmstudio/__init__.py]]"
  - "[[src/mllm_shap/backends/lmstudio/backend.py]]"
  - "[[src/mllm_shap/backends/lmstudio/chat.py]]"
  - "[[src/mllm_shap/backends/lmstudio/models.py]]"
  - "[[src/mllm_shap/backends/lmstudio/options.py]]"
  - "[[src/mllm_shap/backends/mock/__init__.py]]"
  - "[[src/mllm_shap/backends/mock/audio.py]]"
  - "[[src/mllm_shap/backends/mock/backend.py]]"
  - "[[src/mllm_shap/backends/transformers/__init__.py]]"
  - "[[src/mllm_shap/backends/transformers/_torch.py]]"
  - "[[src/mllm_shap/backends/transformers/audio.py]]"
  - "[[src/mllm_shap/backends/transformers/chat.py]]"
  - "[[src/mllm_shap/backends/transformers/embeddings.py]]"
  - "[[src/mllm_shap/backends/transformers/options.py]]"
  - "[[src/mllm_shap/backends/transformers/text.py]]"
---

# backends

## Responsibility

- Generate assistant text for a rendered snapshot, audio parts included where the model hears them, and return a `GenerationRecord`; optionally reuse prompt-prefix state and compute model embeddings.

## Public surface

- All backends take a `PluginContext` and read `context.config.backend`; `generate(snapshot, generation: GenerationConfig)`.
- `MockBackend` (`mock`): deterministic text from a SHA-256 of the snapshot id, each message's `message_chat_text`, model id and generation parameters; at most 16 four-character tokens; used in continuous integration (CI).
- `MockAudioBackend` (`mock_audio`, subclass of `MockBackend`): echoes each message's words, then one token per voiced run of each `AudioPart` (amplitude above `SILENCE` 0.01, runs split by `MIN_GAP_S` 0.01 s) ([[Backends decisions#2026-10-11 — Audio mock backend answers one token per voiced run|decision]]). 16-bit WAV decodes with `wave`; other formats use `decode_audio` (`audio` extra).
- `ApiBackend` (`api`): OpenAI-compatible application programming interface (API) `/chat/completions` over `httpx`, with `Timeout(backend.timeout_s)` and at most `max_inflight` connections.
  - Retries `RETRYABLE_STATUS` (408, 429, 5xx gateway codes) and transport errors up to `backend.max_retries` with exponential backoff, honouring `Retry-After` up to `timeout_s`. Non-JSON responses raise `BackendUnavailableError`.
  - `snapshot_to_api_messages`: a message with audio sends its text part, then one base64 WAV `input_audio` part per `AudioPart` (non-WAV re-encoded).
  - Sends `"logprobs": true`; `extract_logprobs` reads `choices[0].logprobs.content[*].logprob`, else `None`; an empty reply gets `()` (LM Studio sends `null`), which `logprob` scores 0. `returns_logprobs` is true (`SupportsLogprobs`).
  - Caches temperature-0 records (text and logprobs) in an LRU of `RESPONSE_CACHE_SIZE` (1024).
  - `backend.api_host` is required (`ConfigError` without it; no environment or localhost fallback).
  - Bearer key `MLLM_SHAP_API_KEY` for any host; `OPENAI_API_KEY` only for `api.openai.com`.
- `LmStudioBackend` (`lmstudio`): local LM Studio server through the `lmstudio` SDK.
  - `LmStudioOptions` (`hub_search`, `quantization`, `auto_download`, `serialize_generate`, `audio`) come from `backend.options`; unknown keys raise `ConfigError`.
  - `audio: refuse` (default) raises `ValueError` on audio; `transcript` sends the text parts and warns once. The SDK (1.6.0b1 included) takes no audio.
  - `match_downloaded_model_key` matches exactly, then by last-segment prefix, then by substring, and raises on ambiguity; `pick_search_result` raises unless a hub hit matches the name.
  - Retries once, only on transient connection errors (`_is_transient`).
  - Records carry no logprobs (the SDK exposes none).
- `TransformersTextBackend` (`transformers`): Hugging Face model on CUDA (Compute Unified Device Architecture), MPS (Metal Performance Shaders) or CPU.
  - `TransformersOptions`: `dtype` (`auto` picks bfloat16 where supported, float16 on other accelerators, float32 on CPU), `revision` and `model_class`.
  - A config in `MODEL_FOR_CAUSAL_LM_MAPPING` loads as text; any other (or `model_class`) loads its class with `AutoProcessor` ([[Backends decisions#2026-10-11 — Audio input through the transformers processor|decision]]).
  - `audio.processor_inputs` resamples audio to the processor rate and drops empty parts. It sends structured parts when the chat template mentions audio, else one `audio_token` per part.
  - Encoding, generation and decoding share one model lock.
  - With the key-value (KV) cache enabled (`SupportsKvCache`), `_attach_prefix_cache` copies and crops the longest cached prefix and runs only the uncovered tokens.; `kv_cache_hits` and `kv_cache_reused_tokens` count the reuse.
  - Static and contextual embeddings are computed only when `generation.compute_embeddings` is set.
  - `generate` returns raw logits (`output_logits`); `_token_logprobs` takes their log-softmax at each generated token, end-of-sequence included. `returns_logprobs` is true.
- `backends/_chat.py`: `snapshot_has_audio`, `chat_role`, `message_chat_text` (`"<speaker>: <text>"` unless the text is empty; [[Domain decisions#2026-10-08 — Speaker rendered into the message content|speaker rendering]]), `snapshot_to_chat_messages`.

## Data flow

- In: rendered `ConversationSnapshot` and `GenerationConfig` from the [[pipeline]] orchestrator; `PluginContext` at construction.
- Out: `GenerationRecord` built with `value.generation_records.generation_record_from_text(text)` ([[value]]), with model embeddings when requested; other backends have none, so their records equal those `reanalyze` builds.
- Optional dependencies (`httpx`, `lmstudio`, `torch`, `torchaudio`, `transformers`) are imported lazily; a missing extra raises `BackendUnavailableError` ([[core]]).

## Invariants

- Audio prompts skip the prefix cache; a text model and `lmstudio` (unless `audio: transcript`) refuse audio with `ValueError`; `mock` hashes text and the snapshot id only.
- `mock_audio`: a run's token depends on its samples only, so silencing or cutting another run leaves it unchanged.
- In float32, greedy transformers outputs are identical with the prefix cache on and off, and their logprobs agree within 1e-4; in float16 and bfloat16 they agree up to rounding, and the backend warns once ([[Backends decisions#2026-10-09 — Half-precision prefix cache warns instead of promising equality|decision]]). No prompt token is processed twice.
- A backend injected into `ExplainRunner` is reused across calls and closed by the runner; otherwise one backend is built and closed per `explain()`.

## Known issues

- None open. Fixed: C06, C08, C15, D02, D04, D05, D07 ([[Fixed issues]]).

## Tests

- [[tests/backends/test_mock.py]]
- [[tests/backends/test_mock_audio.py]]
- [[tests/backends/test_api.py]]
- [[tests/backends/test_lmstudio.py]]
- [[tests/backends/test_lmstudio_chat.py]]
- [[tests/backends/test_lmstudio_models.py]]
- [[tests/backends/test_lmstudio_options.py]]
- [[tests/backends/test_lmstudio_resolve.py]]
- [[tests/backends/test_transformers_chat.py]]
- [[tests/backends/test_transformers_text.py]]
- [[tests/backends/test_transformers_audio.py]]
- [[tests/backends/test_transformers_gpu.py]]
- [[tests/benchmarks/test_api_backend_bench.py]]
- [[tests/benchmarks/test_mock_audio_bench.py]]
