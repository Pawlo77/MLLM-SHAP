---
title: "Debugging a wrong attribution"
type: handbook
domain: estimation
parent: "[[mllm-shap]]"
tags: [handbook, debugging]
created: 2026-10-07
updated: 2026-10-08
---
# Debugging a wrong attribution

Background: [[How a run works]], `docs/theory/shapley_values.rst`.

## 1. Check efficiency

- For the Shapley estimand with the `identity` normaliser, `sum(values)` must equal `v(N) - v(∅)`.
- The result carries both: `output.result.grand_value - output.result.base_value`. The identity is exact on the full table (`budget.fraction: 1.0`); partial budgets are close (complementary sampling within a few percent on additive games, [[Fixed issues|C12]]). Runs made with 0.1.16 or earlier fail it for every estimator ([[Fixed issues]] C01, C03).
- A normaliser other than `identity` breaks the identity by design; compare raw values.

## 2. Run the axiom probe

- `uv run python .agents/skills/bugspot/scripts/probe_axioms.py` runs all four estimators on 1 to 6 tokens with a counting value function and prints `PASS` or `FAIL` for efficiency and additive-game recovery.

## 3. Inspect the archive

- Each run with an archive path writes `runs/<run_id>/` ([[Run archive]]).
- `sqlite3 runs/<run_id>/archive.sqlite "select count(*), count(distinct coalition_key) from coalition_records"`: more rows than keys means an appended re-run ([[Fixed issues|C02]]).
- `cat runs/<run_id>/base_generation.txt` shows the base generation; `sqlite3 runs/<run_id>/archive.sqlite "select generation_text from coalition_records"` shows what the model produced per coalition (the text is stored in the SQLite row).
- `manifest.json` names the estimand, the estimator and the full configuration ([[Manifest]]); `mllm-shap reanalyze` reproduces a run from it.

## 4. Compare estimators on one archive

- `ExplainRunner(config).reanalyze("runs/<run_id>")` re-scores the archived generations with the runner's value function, estimand and normaliser, without calling the model ([[Reanalyze]]).
- For text-based value functions `reanalyze` reproduces `explain`; embedding values of a transformers run differ, because archives store text only ([[Fixed issues|C08]]).

## 5. Suspect the game, not the estimator

- Under `delete`, distinct masks can render the same prompt (repeated tokens); the [[masking|Absence policy]] and [[value|Value function]] define the game the estimator explains.
- Attributions must not change with `kv_cache.enabled`; if they do, payoffs are misaligned with masks again ([[Fixed issues|C11]]).
- `logprob` ignores the base generation and scores the coalition's own mean token logprob, so a coalition can score high while answering something else ([[Value decisions#2026-10-08 — Logprob value is the coalition's own mean token logprob|Logprob value]]).
