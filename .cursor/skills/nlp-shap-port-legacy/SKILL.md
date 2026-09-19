---
name: nlp-shap-port-legacy
description: >-
  Port a minimal logic slice from legacy MLLM-Shap into nlp_shap. Use when the
  user asks to port, lift, or adapt code from MLLM-Shap / mllm_shap, or to
  compare behavior against the legacy package.
disable-model-invocation: true
---

# Port from MLLM-Shap

## Source

Legacy is a **sibling clone**, not inside this repo:

```text
<parent>/nlp-shap/          # this package
<parent>/MLLM-Shap/         # git clone https://github.com/Pawlo77/MLLM-Shap.git
```

Legacy package root: `../MLLM-Shap/mllm_shap/`

Open `<parent>/nlp-shap/nlp-shap.code-workspace` (see root `AGENTS.md`) so Cursor sees both folders.

## Rules

1. **Minimum slice** — read the legacy module, rewrite into `src/nlp_shap/` matching current package patterns
2. **Never bulk-copy** files, god-classes, experiments, or multimodal stacks unless the task requires it
3. **No hierarchical Shapley** — `shap/hierarchical/*` is dropped; do not port
4. **TDD** — failing test in `tests/` first (mirror `src/` layout)
5. **Imports** — relative inside `src/nlp_shap/`; absolute `from nlp_shap...` in tests/examples
6. **Extras** — torch / transformers / liquid-audio stay optional extras

## Steps

1. Locate the legacy symbol/module under `../MLLM-Shap/mllm_shap/`
2. Write a failing behavior test against the desired `nlp_shap` API
3. Implement the smallest typed rewrite that passes
4. Run `make agent-check`; add docs/notebook only if the public API grew (`docs.mdc`, `examples.mdc`)
5. Hand off per `git-commits.mdc` — no commit unless asked
