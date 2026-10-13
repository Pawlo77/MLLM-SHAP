---
name: rewrite
description: >-
  Use for audio-track work: multimodal or audio modules, SGPA alignment, audio
  backends, or a named rewrite phase.
---

# mllm-shap rewrite (audio track)

The text and audio tracks are done (Phases 0 to 21, released through v0.6.0); new
audio work is ordinary feature work. This skill covers audio and multimodal work only.

## External references (siblings, not in this repository)

Required layout (clone commands: skill `agent-setup`; workspace file
`MLLM-SHAP.code-workspace`):

```text
<parent>/MLLM-SHAP/              # this package
<parent>/MLLM-Shap-legacy/             # git clone https://github.com/Pawlo77/MLLM-Shap-legacy.git
```

| What | Path from `MLLM-SHAP/` |
|------|----------------------|
| Canonical plan | the private research repository, cloned as `../mllm-xai/` (`plans/infrastructure/2026-10-01-mllm-shap-package-rewrite.md`); it depends on this package, never the reverse |
| Legacy reference | `../MLLM-Shap-legacy/mllm_shap/src/mllm_shap/`: gather logic, rewrite; never bulk-copy |

Open the multi-root workspace file before audio-track or port work.

## Principles

1. **Gather logic, rewrite modules**: reference legacy; never bulk-copy files or god-classes
2. **One phase, one tag, one PyPI publish**: do not batch phases
3. **Dropped:** `shap/hierarchical/*`; do not implement or reference
4. **Audio track** (Phases 15 to 21; all done)
5. **Deps:** torch / torchaudio / transformers in extras only
6. **Imports:** relative inside `src/mllm_shap/`; absolute `from mllm_shap...` in tests and examples
7. **No `from __future__ import annotations`** unless required (`python-types.mdc`)

## Current phase gate

This skill is the only place the roadmap lives (session hooks no longer carry it). Read
the plan from the research repository (table above) and compare it with the
`pyproject.toml` version to find the first unchecked **audio** phase. If the sibling is
missing, ask the user to clone it (skill `agent-setup`). Implement **only that
phase**, then follow skill `development` (validate and hand off: docs, notebooks, vault) and
skill `release` for the version bump, tag and publish.

## Audio track order

| Phase | Tag | Module focus |
|-------|-----|--------------|
| 15 | 0.1.18 | multimodal domain (content parts; done) |
| 16 | v0.2.0 | SGPA alignment (Wav2Vec2; done) |
| 17 | v0.2.1 | audio players `sgpa_segments` (done) |
| 18 | v0.4.0 | mock audio backend `mock_audio` (done) |
| 19 | v0.4.1 | audio input: transformers processor, `api` `input_audio`, `lmstudio` transcript (done) |
| 20 | v0.5.0 | audio E2E: snapshot JSON audio + CLI, `transcript: align_only` (done) |
| 21 | v0.6.0 | `loo`, `reference_wer`, `deletion_curve` (done) |

## Workflow

Use skill `development`: red, green, document, refactor, validate, hand off.

User-facing `docs/` describe the shipped API only: no phase numbers or rewrite progress.
See `docs.mdc`.

LM Studio: `pytest -m lms` locally. CI: mock only. GPU: `pytest -m gpu` optional.
