---
title: "Reanalyze"
type: concept
domain: pipeline
tags: [concept]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/nlp_shap/pipeline/reanalyze.py]]"
  - "[[src/nlp_shap/pipeline/runner.py]]"
---

# Reanalyze

Reanalyze re-scores and re-aggregates an existing [[Run archive]] with the current value function, estimand and normaliser, without calling the backend (`ExplainRunner.reanalyze(path)`, `nlp-shap reanalyze`). It re-aggregates with the archived estimator, so it reproduces `explain` for the same text-based value function, and it refuses a different estimator, absence policy or partition and archives of unfinished runs.

Fixed issue: [[Fixed issues|C08]]; decision: [[Pipeline decisions#2026-10-07 — Manifest schema version 2|Manifest schema version 2]].
