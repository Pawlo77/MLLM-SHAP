# Port

Port a minimal slice from legacy MLLM-Shap (skill `nlp-shap-port-legacy`):

1. Confirm sibling layout + workspace (root `AGENTS.md` / `nlp-shap.code-workspace`)
2. Locate the legacy symbol; do **not** bulk-copy
3. Failing test in `tests/` first
4. Smallest typed rewrite in `src/nlp_shap/`
5. `make agent-check` (+ docs/notebook if public API grew)
