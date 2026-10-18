---
title: "Estimation decisions"
type: decision-log
domain: estimation
parent: "[[mllm-shap]]"
created: 2026-10-07
updated: 2026-10-18
tags: [decisions, estimation]
related-code:
  - "[[src/mllm_shap/estimation/complementary.py]]"
  - "[[src/mllm_shap/estimation/neyman.py]]"
  - "[[src/mllm_shap/estimation/_shared.py]]"
  - "[[src/mllm_shap/estimation/exact.py]]"
  - "[[src/mllm_shap/estimation/leverage.py]]"
  - "[[src/mllm_shap/estimation/proxy.py]]"
  - "[[src/mllm_shap/estimation/estimands/_weights.py]]"
  - "[[src/mllm_shap/pipeline/orchestrator.py]]"
---
# Estimation decisions

Architecture: [[estimation]]. Theory: `docs/theory/approximation.rst`.

## 2026-10-18 — Stratified aggregation for mc

Decision: `mc` and the partial-table path of both aggregators use size-stratified with-and-without means over every sampled coalition; `mc` draws sizes uniformly after evaluating sizes `0, 1, n-1, n`. Neyman weights divide by `k` and `n - k`; phase one runs complete passes; complementary sampling adds block pairs per size.

Why: pair means discarded almost every uniform sample once `n >= 10`, and a size-`k` cell's variance scales with `1/k`, not `1/(k+1)`.

Consequences: partial-budget values differ from 0.6.0. Pair means were exact on additive games and the new means are not; elsewhere the new error was lower at `4n` and `32n`. Strata without both means are dropped, so some bias remains below about `16n`. Rejected: imputing missing strata.

## 2026-10-18 — Regression estimators and hierarchical runs

Decision: `leverage`, `kernel_banzhaf` and `proxy` are built in and use only archived masks and payoffs, so `reanalyze` works; `proxy` imports scikit-learn at construction. `explain_hierarchical` runs two games under separate run ids ([[Papers#Musco 2025 — Leverage SHAP|Leverage SHAP]]).

Why: they update every player per evaluation and need no state beyond masks and payoffs.

Consequences: on five ten-player games `proxy` had the lowest Shapley error from `8n`, `kernel_banzhaf` the lowest Banzhaf error from `4n`, and `leverage` beat `complementary` on three games ([[estimation]]). Fine-level values belong to the second game. Rejected: amortised and interaction estimators.

## 2026-10-12 — Leave-one-out bypasses the estimand

Decision: `loo` returns `v(N) - v(N - {i})` from its own payoffs and ignores the configured estimand and budget.

Why: the Spectrogram-Guided Phonetic Alignment ([[SGPA]]) Audio plan (Phase 4) needs leave-one-out as the minimum baseline; it is neither a Shapley nor a Banzhaf value, so routing it through an aggregator would mislabel it.

Consequences: `loo` values need not add up to `v(N) - v(∅)`, and the manifest's estimand describes the run, not the values. Runs past `max_players` set `budget.evaluations`. Rejected: a third estimand (it would also apply to the samplers, which cannot estimate it).

## 2026-10-07 — Singleton-first complementary sampling

Decision: Complementary contribution (CC) sampling and Neyman sampling yield `(N, ∅)`, then the `n` singleton pairs as far as the budget fits, then further pairs; all count toward the budget. Aggregation keeps zero fill for empty strata and warns about them. The Neyman phase-one walk goes size by size ([[Fixed issues|C13]]).

Why: only `({i}, N ∖ {i})` fills player `i`'s size-1 stratum, so partial budgets were biased upwards ([[Fixed issues|C12]]).

Consequences: the efficiency gap at half budget on an additive game dropped from +1.709 to +0.029. Rejected: rescaling by `n / |K|` (worse at every budget); additive imputation of empty strata (error not monotone in the budget).

## 2026-10-07 — Empty and grand pair in CC sampling

Decision: CC and Neyman CC sampling yield `(N, ∅)` exactly once, first. The CC budget is `2^n` masks including the pair. Aggregation pairs masks by complement, not by list position.

Why: Only `(N, ∅)` informs every player's size-`n` stratum and the sampler never drew it, so every estimate was off by `-(v(N) - v(∅)) / n` ([[Fixed issues|C03]], [[Papers#Zhang 2023 — Efficient sampling approaches to Shapley value approximation|Zhang 2023]]).

Consequences: Efficiency holds for CC estimates; a 1-player snapshot returns `v(N) - v(∅)`. The two extra evaluations are boundary coalitions the orchestrator already makes. Partial budgets still starved the size-1 strata until [[Estimation decisions#2026-10-07 — Singleton-first complementary sampling|singleton-first sampling]]. Rejected: documenting the constant offset, which keeps rankings but breaks the `abs_sum` and `min_max` normalisers and every efficiency check.

## 2026-10-07 — Full coalition table

Decision: Exact enumeration evaluates all `2^n` coalitions (`ExactEstimator.num_coalitions(n) == 2^n`). The orchestrator evaluates the empty and grand coalitions for every estimator and seeds `v(N)` from the [[Base generation]] scored against itself. `ExplainResult` carries `base_value` and `grand_value` so users can check efficiency.

Why: Enumeration evaluated `2^n - 1` coalitions, never the grand one, so the marginal `N \ {i} -> N` was missing and efficiency failed ([[Fixed issues|C01]]).

Consequences: One more evaluation per exact run, which takes the vectorised `aggregate_shapley_exact` path. Attributions from 0.1.16 and earlier are not comparable with 0.1.17; the release note says to re-run. Rejected: inferring `v(N)` in the aggregator, which hides the boundary coalition from the archive and [[Reanalyze]].
