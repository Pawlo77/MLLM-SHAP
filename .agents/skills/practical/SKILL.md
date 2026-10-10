---
name: practical
description: >-
  Use when asked to run practical or real-model tests, to check a feature on a real
  model, on /practical <id|glob|all>, or from skill ship when practical_features.py
  prints ids.
---

# nlp-shap practical tests

Scenarios: `tests/practical/scenarios.yaml`; models, variables and VRAM: CONTRIBUTING
"Practical tests"; design: [[Process decisions#2026-10-08 — Practical tests on real small models]]
and [[Process decisions#2026-10-08 — Deep practical tier]].

## 1. Preflight

```bash
uv run python -c "import torch, transformers; print(torch.cuda.is_available() or torch.backends.mps.is_available())"
uv run hf cache ls | grep -E "Qwen2.5-(1.5B|3B)-Instruct|SmolLM2-360M-Instruct"
```

- Missing extras: `uv sync --all-groups --all-extras`.
- No GPU: stop and say so (`NLP_SHAP_PRACTICAL_CPU=1` only if the user asks).
- Weights missing: ask before downloading (quick 3.1 GB and 0.7 GB, deep 6.2 GB); on
  yes, set `NLP_SHAP_PRACTICAL_DOWNLOAD=1` for the run.
- `backend.lmstudio` / `backend.api`: need `lms server start`; they skip without it.
  `make practical` sets `NLP_SHAP_PRACTICAL_LMS_MODEL=qwen2-500m-instruct`.
- Optional quick-tier judge in LM Studio: `NLP_SHAP_PRACTICAL_JUDGE=lmstudio:<key>`.

## 2. Run

```bash
make practical FEATURE='<id|glob|all>'   # quick tier; comma-separated globs allowed
make practical TIER=deep                 # deep.* ids on Qwen2.5-3B, before a release
```

The deep tier runs last with `TIER=all`; loading it frees the quick models (8 GB cap).

## 3. Report

`uv run python -m tests.practical._report` prints `results/latest.json` as one table
per scenario (feature, result, seconds, VRAM) and, for the deep tier, the
error-versus-budget table; add a note column for numbers that matter.

`result` is `passed`, `failed`, `uncertain` (a judge margin under 1 nat) or `skipped`
(with the reason). Add the device, model pins, cache hits and the summary line
`practical: <passed>/<run> passed, <failed> failed, <uncertain> uncertain (<ids>)` for
the PR body.

## 4. A failure

1. Triage with skill `bugspot`: reproduce on the mock backend or a known-truth game
   when possible; a model limit (the model ignores the prompt) is not a library bug.
2. A library bug: write the regression unit test first (`tests/`, no model), then
   the `knowledge/issues/` note (skill `notes`) with "Pinned by" naming the practical
   test id, then the fix.
3. A wrong expectation: change `scenarios.yaml` with the measured numbers in a
   comment; never loosen a threshold without them.
