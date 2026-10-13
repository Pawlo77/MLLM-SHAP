---
title: "Run id"
type: concept
domain: pipeline
parent: "[[mllm-shap]]"
tags: [concept]
created: 2026-10-07
updated: 2026-10-08
related-code:
  - "[[src/mllm_shap/pipeline/runner.py]]"
---
# Run id

The run id names one explain run and its archive directory: `{snapshot_id[:16]}-{digest}`, where the digest covers the configuration (without its archive section) and the effective model id, so different configurations never share an archive and a re-run of the same configuration finds its own.

Decision: [[Pipeline decisions#2026-10-07 — Run id config digest and archive policy|Run id config digest and archive policy]]; fixed issue: [[Fixed issues|C02]].
