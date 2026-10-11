# Tests agents

- `test_*.py`; mirror `src/mllm_shap/`
- Absolute `from mllm_shap...`
- Exit criterion: `make check`; `make tests` for a quick pytest run
- Full markers: `make tests-all`
- `lms` / `gpu` / `bench` are local/optional
- `practical`: real models on a GPU; `make practical [TIER=quick|deep|all] FEATURE=<glob>`, skill `practical`

See `.cursor/rules/tests.mdc` and `tdd.mdc`.
