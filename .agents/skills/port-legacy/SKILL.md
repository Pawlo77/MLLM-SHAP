---
name: port-legacy
description: >-
  Use when asked to port, lift or adapt code from MLLM-Shap-legacy (mllm_shap), or to compare
  behaviour against the legacy package.
---

# Port from MLLM-Shap-legacy

## Source

Legacy is a **sibling clone**, not inside this repo:

```text
<parent>/MLLM-SHAP/          # this package
<parent>/MLLM-Shap-legacy/         # git clone https://github.com/Pawlo77/MLLM-Shap-legacy.git
```

Legacy package root: `../MLLM-Shap-legacy/mllm_shap/src/mllm_shap/`

Open `<parent>/MLLM-SHAP/MLLM-SHAP.code-workspace` (clone commands: `CONTRIBUTING.md`, "Agent setup") so Cursor sees both folders.

## Rules

1. **Minimum slice** — read the legacy module, rewrite into `src/mllm_shap/` matching current package patterns
2. **Never bulk-copy** files, god-classes, experiments, or multimodal stacks unless the task requires it
3. **No hierarchical Shapley** — `shap/hierarchical/*` is dropped; do not port
4. **TDD** — failing test in `tests/` first (mirror `src/` layout)
5. **Imports** — relative inside `src/mllm_shap/`; absolute `from mllm_shap...` in tests/examples
6. **Extras** — torch / transformers / liquid-audio stay optional extras

## Steps

1. Locate the legacy symbol/module under `../MLLM-Shap-legacy/mllm_shap/src/mllm_shap/`
2. Write a failing behavior test against the desired `mllm_shap` API
3. Implement the smallest typed rewrite that passes
4. Run `make check`; add docs and a notebook only if the public API grew (`docs.mdc`, `examples.mdc`)
5. Hand off per `git-commits.mdc` — no commit unless asked
