---
title: "Base generation"
type: concept
domain: pipeline
parent: "[[mllm-shap]]"
tags: [concept]
created: 2026-10-07
related-code:
  - "[[src/mllm_shap/pipeline/orchestrator.py]]"
  - "[[src/mllm_shap/runtime/archive.py]]"
---

# Base generation

The base generation is the model's output for the unmasked prompt, the grand coalition. Every coalition is scored against it, so it is generated first and written to the archive as `base_generation.txt` before any coalition row; its self-score is the grand-coalition payoff.

Fixed issue: [[Fixed issues|C01]]. Walkthrough: [[How a run works]].
