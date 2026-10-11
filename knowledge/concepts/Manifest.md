---
title: "Manifest"
type: concept
domain: pipeline
tags: [concept]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/mllm_shap/pipeline/manifest.py]]"
---

# Manifest

The manifest is `manifest.json` at the root of a [[Run archive]]: schema version (3 since 0.1.18), package version, run and snapshot ids, estimand, estimator, value function, normaliser, absence policy, partition, backend, model, player ids and the full configuration. [[Reanalyze]] reads it to re-aggregate the archive the way it was produced and refuses any other schema version.

Decision: [[Pipeline decisions#2026-10-07 — Manifest schema version 2|Manifest schema version 2]].
