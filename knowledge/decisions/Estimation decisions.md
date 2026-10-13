---
title: "Estimation decisions"
type: decision-log
domain: estimation
parent: "[[mllm-shap]]"
created: 2026-10-07
updated: 2026-10-12
tags: [decisions, estimation]
related-code:
  - "[[src/mllm_shap/estimation/complementary.py]]"
  - "[[src/mllm_shap/estimation/neyman.py]]"
  - "[[src/mllm_shap/estimation/_shared.py]]"
  - "[[src/mllm_shap/estimation/exact.py]]"
  - "[[src/mllm_shap/estimation/estimands/_weights.py]]"
  - "[[src/mllm_shap/pipeline/orchestrator.py]]"
---
# Estimation decisions

Architecture: [[estimation]]. Theory: `docs/theory/approximation.rst`.

## 2026-10-12 — Leave-one-out bypasses the estimand

Decision: `loo` returns `v(N) - v(N - {i})` from its own payoffs and ignores the configured estimand and budget.

Why: the Spectrogram-Guided Phonetic Alignment ([[SGPA]]) Audio plan (Phase 4) needs leave-one-out as the minimum baseline; it is neither a Shapley nor a Banzhaf value, so routing it through an aggregator would mislabel it.

Consequences: `loo` values need not add up to `v(N) - v(∅)`, and the manifest's estimand describes the run, not the values. Runs past `max_players` set `budget.evaluations`. Rejected: a third estimand (it would also apply to the samplers, which cannot estimate it).

## 2026-10-07 — Singleton-first complementary sampling

Decision: Complementary and Neyman sampling yield `(N, ∅)`, then the `n` singleton pairs as far as the budget fits, then random pairs (Neyman: its grid walk); all count toward the budget. Aggregation keeps zero fill for empty strata. The Neyman phase-one walk goes size by size, round-robin over players ([[Fixed issues|C13]]).

Why: Complementary contribution (CC) aggregation counts empty strata as zero, and only `({i}, N ∖ {i})` fills player `i`'s size-1 stratum, so partial budgets were biased upwards ([[Fixed issues|C12]]).

Consequences: On an 8-player game over 200 seeds at fraction 0.1, mean relative error was 0.586 (old), 0.550 (singleton-first) and 0.615 (size-uniform); singleton-first was best or close at every budget and exact at 1.0. The efficiency gap at half budget on an additive game drops from +1.709 to +0.029. At the minimum `2n` masks one player's size-1 stratum can stay empty. Rejected: rescaling by `n / |K|` (worse at every budget); additive imputation of empty strata (error not monotone in the budget).

## 2026-10-07 — Empty and grand pair in CC sampling

Decision: CC and Neyman CC sampling yield `(N, ∅)` exactly once, first. The CC budget is `2^n` masks including the pair. Aggregation pairs masks by complement, not by list position.

Why: Only `(N, ∅)` informs every player's size-`n` stratum and the sampler never drew it, so every estimate was off by `-(v(N) - v(∅)) / n` ([[Fixed issues|C03]], [[Papers#Zhang 2023 — Efficient sampling approaches to Shapley value approximation|Zhang 2023]]).

Consequences: Efficiency holds for CC estimates; a 1-player snapshot returns `v(N) - v(∅)`. The two extra evaluations are boundary coalitions the orchestrator already makes. Partial budgets still starved the size-1 strata until [[Estimation decisions#2026-10-07 — Singleton-first complementary sampling|singleton-first sampling]]. Rejected: documenting the constant offset, which keeps rankings but breaks the `abs_sum` and `min_max` normalisers and every efficiency check.

## 2026-10-07 — Stratified mean for partial samples

Decision: For player `i` and size `k`, estimate the mean marginal `E_k[v(S ∪ {i}) - v(S)]` from the observed pairs of that size. The estimate is the mean of those stratum means weighted by the semivalue's stratum probabilities (`1/n` per size for Shapley, `C(n-1, k) / 2^(n-1)` for Banzhaf), renormalised over observed sizes. `aggregate_from_marginals` keeps its signature.

Why: Summing weighted marginals over the observed pairs of a partial table shrinks every estimate towards zero ([[Fixed issues|C05]]).

Consequences: Consistent, and exact on additive games at any coverage ([[Papers#Castro 2009 — Polynomial calculation of the Shapley value|Castro 2009]], [[Papers#Mitchell 2022 — Sampling permutations for Shapley value estimation|Mitchell 2022]]); not unbiased for general games at low coverage, as the theory page says. 0.1.17 rescaled by `n / observed_strata` with a plain Banzhaf mean; 0.1.18 renormalises the weights. Rejected: the un-normalised sum (biased towards zero); documenting Monte Carlo as inconsistent (leaves the default sampler unusable for partial budgets).

## 2026-10-07 — Full coalition table

Decision: Exact enumeration evaluates all `2^n` coalitions (`ExactEstimator.num_coalitions(n) == 2^n`). The orchestrator evaluates the empty and grand coalitions for every estimator and seeds `v(N)` from the [[Base generation]] scored against itself. `ExplainResult` carries `base_value` and `grand_value` so users can check efficiency.

Why: Enumeration evaluated `2^n - 1` coalitions, never the grand one, so the marginal `N \ {i} -> N` was missing and efficiency failed ([[Fixed issues|C01]]).

Consequences: One more evaluation per exact run, which takes the vectorised `aggregate_shapley_exact` path. Attributions from 0.1.16 and earlier are not comparable with 0.1.17; the release note says to re-run. Rejected: inferring `v(N)` in the aggregator, which hides the boundary coalition from the archive and [[Reanalyze]].
