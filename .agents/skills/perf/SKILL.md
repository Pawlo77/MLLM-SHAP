---
name: perf
description: >-
  Use when a change touches runtime/, estimation/, masking/, pipeline/ or a backend,
  when asked about speed, memory or regressions, or on /bench.
---

# nlp-shap performance

Rules: `.cursor/rules/performance.mdc` (design defaults, end-of-task review) and
`.cursor/rules/benchmarks.mdc` (when a benchmark is required). Benchmarks are local
only; they are not part of `make check` or CI on pull requests.

## When a benchmark is required

- A new backend under `src/nlp_shap/backends/`.
- A change to a hot path: enumeration and sampling (`estimation/`), mask building and
  codecs (`masking/`), scheduling, dedup, caches and archive writes (`runtime/`), or the
  orchestrator loop (`pipeline/`).
- Benchmarks live in `tests/benchmarks/`, mirror the source path, and carry
  `@pytest.mark.bench`.

## Run

```bash
make bench              # all benchmarks (pytest-benchmark); JSON in .pytest_benchmark/benchmark.json
make bench-regression   # fail when a median is >25% slower than tests/benchmarks/baseline.json
make bench-baseline     # re-record tests/benchmarks/baseline.json on this machine
```

## Baselines

- `tests/benchmarks/baseline.json` is a pytest-benchmark save file with the reference
  medians; `tests/benchmarks/AGENTS.md` names the machine that produced it.
- `make bench-regression` compares with `--benchmark-compare=tests/benchmarks/baseline.json
  --benchmark-compare-fail=median:25%`; `make bench-baseline` re-records the file.
- Benchmarks use the `benchmark` fixture (`benchmark(fn, *args)`) and keep correctness
  asserts on the returned value.
- Regenerate baselines only on the maintainer's machine, in a commit of its own
  (`perf: refresh benchmark baselines`), and say why in the body.

## Memory probe

```python
import tracemalloc

tracemalloc.start()
output = runner.explain_sync(snapshot)  # the operation under test
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"peak {peak / 2**20:.1f} MiB for n={len(output.result.values)} players")
```

Run it for n and 2n players. Peak memory that doubles per added player means `2^n`
materialisation: stream coalitions or batch the jobs instead.

## Review questions

1. What is the time and memory complexity in players `n` and budget `m`?
2. Is anything materialised per coalition that could be streamed?
3. Is a prompt re-tokenised per mask instead of once per snapshot?
4. Does the scheduler create one coroutine per job up front instead of bounding
   in-flight work?
5. Did `make bench-regression` pass, and is a new benchmark needed?
