---
name: nlp-shap-rewrite
description: >-
  Audio-track and legacy-port phases for nlp_shap after text MVP (v0.1.x).
  Use when implementing multimodal/audio modules (phases 15–21), SGPA alignment,
  liquid-audio, or when the user names a rewrite phase / audio track.
disable-model-invocation: true
---

# nlp-shap rewrite (audio track)

Text track (phases 0–14, through `v0.1.15`) is **done**. Current package is past
that baseline — check `pyproject.toml` version. This skill covers **audio /
multimodal** work only.

## External references (siblings — not in this repo)

Required layout — see root `AGENTS.md` / `nlp-shap.code-workspace`:

```text
<parent>/nlp-shap/              # this package
<parent>/nlp-shap-research/     # git clone https://github.com/Pawlo77/nlp-shap-research.git
<parent>/MLLM-Shap/             # git clone https://github.com/Pawlo77/MLLM-Shap.git
```

| What | Path from `nlp-shap/` |
|------|----------------------|
| Canonical plan | `../nlp-shap-research/docs/plans/infrastructure/` (Obsidian: nlp-shap Package Rewrite) |
| Legacy reference | `../MLLM-Shap/mllm_shap/` — gather logic, rewrite; never bulk-copy |

Open the multi-root workspace file before audio-track / port work.

## Principles

1. **Gather logic, rewrite modules** — reference legacy; never bulk-copy files or god-classes
2. **One phase = one tag = one PyPI publish** — do not batch phases
3. **Dropped:** `shap/hierarchical/*` — do not implement or reference
4. **Audio track** (Phases 15–21, `v0.2.0`–`v0.2.6`) only after text sign-off (already done)
5. **Deps:** torch / transformers / liquid-audio in extras only
6. **Imports:** relative inside `src/nlp_shap/`; absolute `from nlp_shap...` in tests and examples
7. **No `from __future__ import annotations`** unless required — see `python-types.mdc`

## Current phase gate

Check the plan's first unchecked **audio** phase. Implement **only that phase**, then
follow skill `nlp-shap-development` validate + hand-off (docs, notebooks, tag, publish).

## Audio track order (v0.2.x)

| Phase | Tag | Module focus |
|-------|-----|--------------|
| 15 | v0.2.0 | multimodal domain + enums |
| 16 | v0.2.1 | SGPA alignment (Wav2Vec2) |
| 17 | v0.2.2 | audio masking, filters, sgpa partition |
| 18 | v0.2.3 | mock audio backend |
| 19 | v0.2.4 | liquid-audio backend |
| 20 | v0.2.5 | multimodal E2E pipeline |
| 21 | v0.2.6 | audio value fns + SGPA Audio smoke |

## Workflow

Use `nlp-shap-development`: red → green → document → refactor → validate → hand off.

User-facing `docs/` describe the shipped API only — no phase numbers or rewrite progress.
See `docs.mdc`.

LM Studio: `pytest -m lms` locally. CI: mock only. GPU: `pytest -m gpu` optional.
