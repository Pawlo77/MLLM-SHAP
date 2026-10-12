# Check: MLLM-SHAP

`make check` runs `make agent-check` (filtered pytest with
`not lms and not gpu and not bench and not slow and not practical`, then `prek-all`), then
`make vault-check`. Optional full marker run: `make tests-all`.

| Changed | Also run |
|---------|----------|
| `docs/` or public API | `make docs` |
| `examples/*.ipynb`, a `src/` change to a public workflow, or a release PR | `make notebooks` |
| dependencies or `pyproject.toml` | `uv lock`, then `make check` |
| hot path or backend | `make bench` (local); `make bench-regression` against `tests/benchmarks/baseline.json` |
| `src/mllm_shap/<pkg>/`, docstrings, `knowledge/`, `plans/` | `make knowledge vault-check`; architecture note updated (skill `notes`) |
