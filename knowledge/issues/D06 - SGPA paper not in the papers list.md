---
title: "D06 - SGPA paper not in the papers list"
type: issue
domain: alignment
parent: "[[mllm-shap]]"
status: deferred
severity: minor
found: 2026-10-10
created: 2026-10-10
fixed_in: ""
tags: [issue, deferred, papers]
related-code:
  - "[[src/mllm_shap/alignment/sgpa.py]]"
---

# D06 - SGPA paper not in the papers list

Package: [[alignment]]. Concept: [[SGPA]].

## Symptom

- [[alignment]] implements [[SGPA]] (arXiv 2603.02250), but [[Papers]] has no section for it and no paper-to-code compliance table, as skill `notes` requires for an implemented method.

## Cause

- The paper is under review at a conference; its text and claims may still change, so a section and a compliance table written now could describe a different version.

## Repro

```bash
grep -n "2603.02250" knowledge/Papers.md   # no match
```

## Fix

- Deferred until the paper is accepted or the review ends: add its section to [[Papers]] (skill `notes`, procedure "paper") and run the `methodologist` agent for the compliance table against `alignment/ctc.py` and `alignment/sgpa.py`.
