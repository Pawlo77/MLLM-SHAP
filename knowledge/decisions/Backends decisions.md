---
title: "Backends decisions"
type: decision-log
domain: backends
created: 2026-10-07
updated: 2026-10-08
tags: [decisions, backends]
related-code:
  - "[[src/nlp_shap/protocols/value.py]]"
  - "[[src/nlp_shap/pipeline/orchestrator.py]]"
  - "[[src/nlp_shap/backends/transformers/text.py]]"
---

# Backends decisions

Part of [[nlp-shap]]; architecture: [[backends]]. Related: [[value]], [[Capability protocol]].

## 2026-10-07 — Embeddings computed only when the value function needs them

Decision: Value functions that score embeddings declare it through the `RequiresEmbeddings` capability (`needs_embeddings`). The orchestrator then sets `GenerationConfig.compute_embeddings`, backends compute embeddings only when it is set, and the orchestrator swaps the backend's embeddings into the text-derived scoring record.

Why: The transformers backend ran two extra forward passes per coalition (static and contextual embeddings) under the model lock, and the pipeline then scored hash embeddings of the text, discarding the work.

Consequences: Text-based runs (the default `tfidf_cosine`) skip both passes. A transformers run's embedding values differ between `explain` (model embeddings) and `reanalyze` (hash embeddings of archived text), because archives store text only; the 0.1.18 release notes say so. Rejected: always computing embeddings, which pays two passes per coalition for value functions that never read them.
