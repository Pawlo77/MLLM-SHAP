---
title: "D02 - Audio backend"
type: issue
domain: backends
status: deferred
severity: major
found: 2026-10-07
created: 2026-10-07
updated: 2026-10-11
fixed_in: ""
tags: [issue, deferred, audio]
related-code:
  - "[[src/mllm_shap/domain/conversation.py]]"
  - "[[src/mllm_shap/backends/transformers/chat.py]]"
  - "[[src/mllm_shap/masking/policies.py]]"
  - "[[src/mllm_shap/viz/labels.py]]"
---

# D02 - Audio backend

Packages: [[backends]], [[domain]], [[masking]], [[viz]]. Roadmap: [[Roadmap]].

## Symptom

- Messages can carry `AudioPart` content since 0.1.18, but the API (application programming interface), LM Studio and transformers backends refuse audio with `ValueError`; only the test backend `mock_audio` reacts to audio, so no model run can explain audio yet.

## Cause

- No model backend implements audio generation yet; `mock_audio` only hashes voiced runs.
- The prerequisites landed in 0.1.18: message content parts, snapshot ids that hash audio bytes, and partitioner-owned player spans read by policies and labels.

## Repro

Explain a snapshot whose message has an `AudioPart` with the transformers or API backend: both raise `ValueError`.

## Fix

- Deferred to the audio track (versions 0.2.x, [[Roadmap]]). Prerequisites done: player spans owned by the partitioner and message content parts (0.1.18), [[SGPA]] alignment (0.2.0), audio players and audio absence rendering (`sgpa_segments`, 0.2.1), the `mock_audio` backend for continuous integration (CI) audio runs (0.4.0, [[Backends decisions#2026-10-11 — Audio mock backend answers one token per voiced run|decision]]).
