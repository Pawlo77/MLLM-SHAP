---
title: "estimation"
type: module
domain: estimation
parent: "[[mllm-shap]]"
tags: [architecture, estimation]
created: 2026-10-07
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/estimation/__init__.py]]"
  - "[[src/mllm_shap/estimation/_shared.py]]"
  - "[[src/mllm_shap/estimation/complementary.py]]"
  - "[[src/mllm_shap/estimation/estimands/__init__.py]]"
  - "[[src/mllm_shap/estimation/estimands/_weights.py]]"
  - "[[src/mllm_shap/estimation/estimands/banzhaf.py]]"
  - "[[src/mllm_shap/estimation/estimands/shapley.py]]"
  - "[[src/mllm_shap/estimation/exact.py]]"
  - "[[src/mllm_shap/estimation/monte_carlo.py]]"
  - "[[src/mllm_shap/estimation/neyman.py]]"
  - "[[src/mllm_shap/estimation/normalizers.py]]"
---

# estimation

## Responsibility

- Choose which coalitions to evaluate (Estimator) and turn coalition payoffs into per-player attributions (Estimand aggregators).
- Rescale attributions for presentation (normalisers).

## Public surface

- `ExactEstimator` (`exact`): yields all `2^n` masks in bitmask order (`num_coalitions(n) == 2^n`); requires `budget_fraction == 1.0` (raised by `sample_masks` and refused earlier by config validation).
- `MonteCarloEstimator` (`mc`): yields the grand and empty coalitions, the empty and singleton coalitions when `include_minimal_masks`, then uniform random masks without repeats until `compute_mc_num_samples` (a fraction of `2^n`, boundary masks included).
- `ComplementaryEstimator` (`complementary`): yields the pair `(N, ∅)`, then all `n` singleton pairs `({i}, N ∖ {i})`, then random complementary pairs (Complementary contribution, [[Estimation decisions#2026-10-07 — Singleton-first complementary sampling|Singleton-first complementary sampling]]); `m_counts` exposes the sampled count matrix.
- `NeymanEstimator` (`neyman_cc`, the default): phase one yields `(N, ∅)` and the singleton pairs, then walks the `(size, player)` grid size by size, round-robin over players, until every cell reaches `initial_num_splits` or its count of distinct coalitions. `begin_allocation(masks, payoffs)` estimates stratum variances and water-fills the remaining pairs over coalition sizes (complement column `n - j`); `sample_allocation_masks()` yields phase two (Neyman allocation). Options come from `explanation.neyman`.
- Every estimator: `estimate_attributions(masks, payoffs, aggregator)`. Exact and Monte Carlo inherit it from `_shared._SnapshotEstimator` (with `options_model`, `__init__` and `bind_snapshot`), which delegates to the aggregator. The complementary estimators pair masks by complement (`pair_masks_by_complement`, `build_c_matrix_from_pairs`, `aggregate_complementary_shapley`) and accept only the Shapley estimand.
- `ShapleyAggregator`, `BanzhafAggregator`: `aggregate(masks, payoffs)`. Shapley sends a complete `2^n` table through the vectorised `aggregate_shapley_exact`. Banzhaf and partial tables go through `aggregate_from_marginals`: the mean observed marginal per coalition size, renormalised over the observed strata with the semivalue's stratum probabilities ([[Estimation decisions#2026-10-07 — Stratified mean for partial samples|Stratified mean for partial samples]]). A player with no observed marginal gets 0, and one `WARNING` lists all such players.
- Normalisers: `IdentityNormalizer`, `AbsSumNormalizer`, `PowerShiftNormalizer` (`power` option), `MinMaxNormalizer`.
- `_shared.py` budget helpers:
  - `mc_coalition_budget(n)` and `complementary_coalition_budget(n)` are `2^n`.
  - `budget_fraction(n, fraction, evaluations)` is the one place an absolute `budget.evaluations` becomes the samplers' fraction: `evaluations / 2^n`, capped at 1, exact because the divisor is a power of two ([[Pipeline decisions#2026-10-08 — Absolute budget passed to samplers as an exact fraction|Absolute budget passed to samplers as an exact fraction]]).
  - `compute_mc_num_samples`: `min(max(floor, int(2^n * fraction)), 2^n)` with floor 2, or `n + 2` with minimal masks.
  - `compute_complementary_num_splits`: `min(max(min(2n + 2, 2^n), t - t % 2), 2^n)` for `t = int(2^n * fraction)`, so always even.
  - `random_present`: uniform over all coalitions, over size-`k` coalitions, or over those containing a fixed player; `ValueError` for an impossible size.

## Data flow

- In: `PlayerSet`, the budget fraction (`budget.fraction`, or `budget.evaluations` converted by `budget_fraction`), `include_minimal_masks`, `seed` from the [[pipeline]] config; later the masks and payoffs the orchestrator collected (both phases for Neyman).
- Out: an iterator of `CoalitionMask` and a `list[float]` of attributions.
- Imports [[domain]], [[protocols]], [[plugins]] (context type only) and numpy.

## Invariants

- On the full table every estimator satisfies efficiency: `sum(values) == v(N) - v(∅)` for the Shapley estimand ([[Estimation decisions#2026-10-07 — Full coalition table|Full coalition table]], [[Estimation decisions#2026-10-07 — Empty and grand pair in CC sampling|Empty and grand pair in CC sampling]]).
- Every sampler yields the grand and empty coalitions first, and they count toward the budget.
- Complementary aggregation does not depend on mask order.
- The complementary and Neyman budget floor `min(2n + 2, 2^n)` fits `(N, ∅)` and every singleton pair, so every size-1 stratum is seeded. At the floor the symmetric game `v(S) = |S|` gives exactly 1 per player ([[Fixed issues|C14]]).
- Partial-budget estimates converge to the exact value as coverage grows; they are exact on additive games for Monte Carlo and close to efficient for complementary sampling ([[Fixed issues|C12]]).

## Known issues

- Fixed: M01, M02, M03, C01, C03, C04, C05, C10, C12, C13, C14 ([[Fixed issues]]).

## Tests

- [[tests/estimation/test_axioms.py]]
- [[tests/estimation/test_exact.py]]
- [[tests/estimation/test_monte_carlo.py]]
- [[tests/estimation/test_complementary.py]]
- [[tests/estimation/test_neyman.py]]
- [[tests/estimation/test_neyman_internals.py]]
- [[tests/estimation/test_shared.py]]
- [[tests/estimation/test_normalizers.py]]
- [[tests/estimation/estimands/test_weights.py]]
- [[tests/estimation/estimands/test_divergence.py]]
- [[tests/benchmarks/test_estimation_bench.py]]
