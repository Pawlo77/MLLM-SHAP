---
title: "Roadmap"
type: handbook
domain: roadmap
tags: [handbook, roadmap]
created: 2026-10-07
updated: 2026-10-11
---

# Roadmap

Part of [[mllm-shap]]. Detailed plans live outside this repository.

## Remediation (2026-10)

| Order | Work | Release |
|---|---|---|
| 1 | Portable agent setup, committed memory, this vault | none |
| 2 | Correctness: efficient attributions for every estimator (C01 to C11 in [[Fixed issues]]) | 0.1.17 |
| 3 | Documentation cleanup, continuous integration (CI) and security hardening | none |
| 4 | Typed plugin contracts, player spans owned by the partition, backend runtime hardening ([[Fixed issues\|C06]]), one release ([[Process decisions#2026-10-07 — Single release 0.1.18 for Phases 3 and 2\|Single release 0.1.18 for Phases 3 and 2]]) | 0.1.18 |
| 5 | Test hygiene, command line, TaskGroup scheduler, mutation testing, sampler fixes ([[Fixed issues\|C12]]) | 0.1.18 |

## Audio track

One phase, one tag, one publish (skill `rewrite`). Pull request (PR) 4 was ported per phase onto the 0.1.18 message model and closed ([[Fixed issues|D03]]).

| Phase | Tag | Module focus |
|---|---|---|
| 15 | 0.1.18 | multimodal domain: message content parts (done) |
| 16 | v0.2.0 | Spectrogram-Guided Phonetic Alignment ([[SGPA]]) with Wav2Vec2 (done) |
| 17 | v0.2.1 | audio players (`sgpa_segments`) and audio absence rendering (done) |
| 18 | 0.4.0 | mock audio backend (`mock_audio`, done) |
| 19 | 0.4.1 | audio input for transformers audio models, the application programming interface (API) backend and LM Studio (done) |
| 20 | next | audio end-to-end pipeline on any audio backend |
| 21 | later | audio value functions and SGPA audio smoke test |

Phase 19 replaced a dedicated liquid-audio backend ([[Backends decisions#2026-10-11 — Audio input through the transformers processor|decision]]).
