# Benchmark agents

Rules: `.cursor/rules/benchmarks.mdc`; procedure: skill `perf`. Only what is specific to this
directory is listed here.

- Build fresh state inside the timed function when it is stateful (registries, archives).
- Name the module after the hot path under `src/mllm_shap/` and point its `@see` tag at it.
- `conftest.py` redacts the host name; re-record `baseline.json` with `make bench-baseline`
  after an intended speed change.

## Baseline machine

`baseline.json` was recorded on 2026-10-07 on an Apple M4 Pro (14 cores, 48 GB RAM),
macOS 26.7 (Darwin 25.6.0), CPython 3.12.12. Compare against
it only on comparable hardware; on another machine record a local baseline first.
