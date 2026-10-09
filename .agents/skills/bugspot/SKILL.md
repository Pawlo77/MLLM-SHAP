---
name: bugspot
description: >-
  Use when asked to find bugs, spot issues or stress test nlp-shap, or on /bugspot.
---

# nlp-shap bugspot

## Probes (run all; each prints PASS/FAIL)

```bash
uv run python .agents/skills/bugspot/scripts/probe_axioms.py
uv run pytest tests/estimation/test_axioms.py tests/pipeline/test_pipeline_axioms.py --hypothesis-seed=0 -q
uv run pytest -p no:randomly -q -x tests/runtime tests/pipeline
```

## Manual probe list

- 1-player, 2-player, 25-player (budget 0.01) snapshots through every estimator.
- Delete policy with repeated tokens; neutral policy with empty messages.
- Backend raising on job k of n with `max_inflight` 1 and 4.
- Re-run into an existing archive; reanalyze with a different value function and
  with a different estimator (must raise).
- `PYTHONHASHSEED=1` vs `2` for every hash in `masking/codec.py`, `runtime/dedup.py`.
- Config with unknown plugin names: error must mention the group and known names.

## Output

Findings as `C<n>: <title> — repro command — file:line — expected vs actual`.
Write a failing regression test for each confirmed finding under `tests/`
before proposing a fix; do not fix in the same step.
