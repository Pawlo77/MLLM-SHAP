---
paths:
  - "tests/**/*.py"
---

<!-- Generated from .cursor/rules/tests.mdc by scripts/sync_rules.py; do not edit. -->

# Tests

- Files: `test_*.py` under `tests/`, mirror `src/mllm_shap/` layout
- Absolute imports: `from mllm_shap...`
- Behavior tests over trivia / mock theater
- Estimator, estimand aggregator, masking or orchestrator changes must pass the axiom tests (`tests/estimation/test_axioms.py`, `tests/pipeline/test_pipeline_axioms.py`) and add a known-truth case for any new code path
- Assert values (known-truth games, exact numbers), not only shapes or types
- Markers: `bench` (local `make bench`), `lms`, `gpu`, `slow` (wheel build or network;
  `make tests-all` only) — skip when deps/host missing
- `practical(<feature>)`: real small model on a GPU, `make practical FEATURE=<glob>` (excluded from `make check`)
- Agent default / `make check`: `pytest -m "not lms and not gpu and not bench and not slow and not practical"` (`make agent-check`)
- Full markers: `make tests-all`
- Hook helper tests (e.g. `tests/hooks/`) OK for tooling outside the package
