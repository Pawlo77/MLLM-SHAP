---
title: "Backends decisions"
type: decision-log
domain: backends
parent: "[[mllm-shap]]"
created: 2026-10-07
updated: 2026-10-11
tags: [decisions, backends]
related-code:
  - "[[src/mllm_shap/protocols/value.py]]"
  - "[[src/mllm_shap/pipeline/orchestrator.py]]"
  - "[[src/mllm_shap/backends/transformers/text.py]]"
  - "[[src/mllm_shap/backends/mock/audio.py]]"
  - "[[src/mllm_shap/backends/transformers/audio.py]]"
---
# Backends decisions

Architecture: [[backends]]. Related: [[value]], [[Capability protocol]].

## 2026-10-11 — Audio input through the transformers processor

Decision: The `transformers` kind hears audio: a checkpoint `AutoModelForCausalLM` does not know loads its own class with `AutoProcessor`. `api` sends `input_audio` parts; `lmstudio` refuses audio unless `audio: transcript`.

Why: A dedicated liquid-audio backend served one model family. Granite Speech, Qwen2-Audio and Voxtral share the processor path, and the LM Studio SDK (1.6.0b1 included) takes no audio.

Consequences: Granite Speech's chat template has no audio slot, so audio goes in as `audio_token`; templates that mention audio get structured parts. Audio prompts skip the prefix cache. `torchaudio` joins the `transformers` extra. Rejected: a `transformers_audio` kind (a second loader for the same models).

## 2026-10-11 — Audio mock backend answers one token per voiced run

Decision: `MockAudioBackend` is a separate `mock_audio` entry, not a mode of `mock`. It echoes message words and adds one token per voiced run of each `AudioPart`, the SHA-256 of the run's 16-bit samples; 16-bit WAV decodes with the standard library `wave`.

Why: `mock` hashes the whole prompt, so audio players got arbitrary values; continuous integration (CI) must run audio explanations without the `audio` extra.

Consequences: silencing or cutting a word's samples drops exactly its token, so audio attributions are checkable. `mock` outputs stay unchanged. Rejected: a flag on `mock` (changes its archived outputs), fixed-length windows (cutting shifts every later window).

## 2026-10-09 — Half-precision prefix cache warns instead of promising equality

Decision: `TransformersTextBackend` keeps the prefix cache in float16 and bfloat16 and logs one `WARNING` per backend that cached results can differ from `kv_cache` off; float32 is the documented setting for identical results.

Why: The cached path splits the prompt over several forward passes and the uncached path uses one; half-precision attention rounds them differently and can flip a near-tied greedy token (D04: up to 0.0138 on Qwen2.5-3B float16 on Apple Metal Performance Shaders (MPS)). The guide claimed equality.

Consequences: Default `dtype: auto` runs on accelerators keep the speed-up with a visible caveat; `test_transformers_backend_warns_once_on_half_precision_prefix_cache` pins it ([[Fixed issues]]). Rejected: disabling the cache in half precision (loses the speed-up on every accelerator); a float32 prefix pass (different kernels, still not bit-identical).

## 2026-10-07 — Embeddings computed only when the value function needs them

Decision: Value functions that score embeddings declare it through the `RequiresEmbeddings` capability (`needs_embeddings`). The orchestrator then sets `GenerationConfig.compute_embeddings`, backends compute embeddings only when it is set, and the orchestrator swaps the backend's embeddings into the text-derived scoring record.

Why: The transformers backend ran two extra forward passes per coalition (static and contextual embeddings) under the model lock, and the pipeline then scored hash embeddings of the text, discarding the work.

Consequences: Text-based runs (the default `tfidf_cosine`) skip both passes. A transformers run's embedding values differ between `explain` (model embeddings) and `reanalyze` (hash embeddings of archived text), because archives store text only; the 0.1.18 release notes say so. Rejected: always computing embeddings, which pays two passes per coalition for value functions that never read them.
