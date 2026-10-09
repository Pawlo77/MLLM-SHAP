---
title: "Roadmap"
type: handbook
domain: roadmap
tags: [handbook, roadmap]
created: 2026-10-07
updated: 2026-10-08
---

# Roadmap

Part of [[nlp-shap]]. Detailed plans live outside this repository.

## Remediation (2026-10)

| Order | Work | Release |
|---|---|---|
| 1 | Portable agent setup, committed memory, this vault | none |
| 2 | Correctness: efficient attributions for every estimator (C01 to C11 in [[Fixed issues]]) | 0.1.17 |
| 3 | Documentation cleanup, continuous integration (CI) and security hardening | none |
| 4 | Typed plugin contracts, player spans owned by the partition, backend runtime hardening ([[Fixed issues\|C06]]), one release ([[Process decisions#2026-10-07 — Single release 0.1.18 for Phases 3 and 2\|Single release 0.1.18 for Phases 3 and 2]]) | 0.1.18 |
| 5 | Test hygiene, command line, TaskGroup scheduler, mutation testing, sampler fixes ([[Fixed issues\|C12]]) | 0.1.18 |

## Audio track

One phase, one tag, one publish (skill `rewrite`); pull request (PR) 4 is rebased and split per phase ([[D03 - PR 4 rebase]]).

| Phase | Tag | Module focus |
|---|---|---|
| 15 | v0.2.0 | multimodal domain and enums |
| 16 | v0.2.1 | Spectrogram-Guided Phonetic Alignment ([[SGPA]]) with Wav2Vec2 |
| 17 | v0.2.2 | audio masking, filters, SGPA partition |
| 18 | v0.2.3 | mock audio backend |
| 19 | v0.2.4 | liquid-audio backend |
| 20 | v0.2.5 | multimodal end-to-end pipeline |
| 21 | v0.2.6 | audio value functions and SGPA audio smoke test |

Prerequisite: [[D02 - Audio backend]].
