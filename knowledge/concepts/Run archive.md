---
title: "Run archive"
type: concept
domain: runtime
tags: [concept]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/mllm_shap/runtime/archive.py]]"
---

# Run archive

The run archive is the directory one explain run persists: `manifest.json`, `base_generation.txt` and an SQLite database `archive.sqlite` (WAL journaling, schema version 4) with one row per coalition evaluation: key, packed mask, policy, model, timing, the generated text and its token logprobs (`NULL` when the backend returned none). A completion marker distinguishes finished runs. It lets [[Reanalyze]] re-score the run without the model.

Identified by the [[Run id]]; described by the [[Manifest]]. Fixed issue: [[Fixed issues|C02]].
