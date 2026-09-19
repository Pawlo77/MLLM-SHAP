# Bench

Run local performance benchmarks (not part of `make check` / CI):

```bash
make bench
```

Optional regression gate vs baselines:

```bash
make bench-regression
```

Use when changing hot paths under `runtime/`, `estimation/`, `masking/`, or backends. See `.cursor/rules/benchmarks.mdc`.
