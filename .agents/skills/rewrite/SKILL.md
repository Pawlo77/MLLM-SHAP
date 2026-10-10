---
name: rewrite
description: >-
  Use for audio-track work: multimodal or audio modules, SGPA alignment, liquid-audio,
  or a named rewrite phase.
---

# nlp-shap rewrite (audio track)

The text track is done; the audio track (0.2.x) is in progress (check `version` in
`pyproject.toml`). This skill covers audio and multimodal work only.

## External references (siblings, not in this repository)

Required layout (clone commands: `CONTRIBUTING.md`, "Agent setup"; workspace file
`nlp-shap.code-workspace`):

```text
<parent>/nlp-shap/              # this package
<parent>/MLLM-Shap/             # git clone https://github.com/Pawlo77/MLLM-Shap.git
```

| What | Path from `nlp-shap/` |
|------|----------------------|
| Canonical plan | the private research repository, cloned as `../nlp-shap-research/` (`docs/plans/infrastructure/`); it depends on this package, never the reverse |
| Legacy reference | `../MLLM-Shap/mllm_shap/src/mllm_shap/`: gather logic, rewrite; never bulk-copy |

Open the multi-root workspace file before audio-track or port work.

## Principles

1. **Gather logic, rewrite modules**: reference legacy; never bulk-copy files or god-classes
2. **One phase, one tag, one PyPI publish**: do not batch phases
3. **Dropped:** `shap/hierarchical/*`; do not implement or reference
4. **Audio track** (Phases 15 to 21; next is Phase 18, `v0.2.2`)
5. **Deps:** torch / transformers / liquid-audio in extras only
6. **Imports:** relative inside `src/nlp_shap/`; absolute `from nlp_shap...` in tests and examples
7. **No `from __future__ import annotations`** unless required (`python-types.mdc`)

## Current phase gate

This skill is the only place the roadmap lives (session hooks no longer carry it). Read
the plan from the research repository (table above) and compare it with the
`pyproject.toml` version to find the first unchecked **audio** phase. If the sibling is
missing, ask the user to clone it (`CONTRIBUTING.md`, "Agent setup"). Implement **only that
phase**, then follow skill `development` (validate and hand off: docs, notebooks, vault) and
skill `release` for the version bump, tag and publish.

## Audio track order (v0.2.x)

| Phase | Tag | Module focus |
|-------|-----|--------------|
| 15 | 0.1.18 | multimodal domain (content parts; done) |
| 16 | v0.2.0 | SGPA alignment (Wav2Vec2; done) |
| 17 | v0.2.1 | audio players `sgpa_segments` (done) |
| 18 | v0.2.2 | mock audio backend |
| 19 | v0.2.3 | liquid-audio backend |
| 20 | v0.2.4 | multimodal E2E pipeline |
| 21 | v0.2.5 | audio value fns + SGPA Audio smoke |

## Workflow

Use skill `development`: red, green, document, refactor, validate, hand off.

User-facing `docs/` describe the shipped API only: no phase numbers or rewrite progress.
See `docs.mdc`.

LM Studio: `pytest -m lms` locally. CI: mock only. GPU: `pytest -m gpu` optional.
