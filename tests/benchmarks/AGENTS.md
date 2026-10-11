# Benchmark agents

Rules: `.cursor/rules/benchmarks.mdc`; procedure: skill `perf`. Only what is specific to this
directory is listed here.

- Build fresh state inside the timed function when it is stateful (registries, archives).
- Name the module after the hot path under `src/mllm_shap/` and point its `@see` tag at it.
- `conftest.py` redacts the host name; re-record `baseline.json` with `make bench-baseline`
  after an intended speed change.
