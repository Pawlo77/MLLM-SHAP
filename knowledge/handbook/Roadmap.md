---
title: "Roadmap"
type: handbook
domain: roadmap
parent: "[[mllm-shap]]"
tags: [handbook, roadmap]
created: 2026-10-07
updated: 2026-10-12
---

# Roadmap

Detailed plans live outside this repository.

## Remediation (2026-10)

| Order | Work | Release |
|---|---|---|
| 1 | Repository tooling and this vault | none |
| 2 | Correctness: efficient attributions for every estimator (C01 to C11 in [[Fixed issues]]) | 0.1.17 |
| 3 | Documentation cleanup, continuous integration (CI) and security hardening | none |
| 4 | Typed plugin contracts, player spans owned by the partition, backend runtime hardening ([[Fixed issues\|C06]]), one release | 0.1.18 |
| 5 | Test hygiene, command line, TaskGroup scheduler, mutation testing, sampler fixes ([[Fixed issues\|C12]]) | 0.1.18 |

## Audio track

Complete: every phase is released. One phase, one tag, one publish (skill `rewrite`). Pull request (PR) 4 was ported per phase onto the 0.1.18 message model and closed ([[Fixed issues|D03]]).

| Phase | Tag | Module focus |
|---|---|---|
| 15 | 0.1.18 | multimodal domain: message content parts (done) |
| 16 | v0.2.0 | Spectrogram-Guided Phonetic Alignment ([[SGPA]]) with Wav2Vec2 (done) |
| 17 | v0.2.1 | audio players (`sgpa_segments`) and audio absence rendering (done) |
| 18 | 0.4.0 | mock audio backend (`mock_audio`, done) |
| 19 | 0.4.1 | audio input for transformers audio models, the application programming interface (API) backend and LM Studio (done) |
| 20 | 0.5.0 | audio end-to-end pipeline: audio in snapshot JSON and the command line, hearing-only `transcript: align_only` (done) |
| 21 | 0.6.0 | research utilities: `loo` estimator, `reference_wer` value function, `deletion_curve` (done; the smoke experiment lives in `mllm-xai`) |

Phase 19 replaced a dedicated liquid-audio backend ([[Backends decisions#2026-10-11 — Audio input through the transformers processor|decision]]).
