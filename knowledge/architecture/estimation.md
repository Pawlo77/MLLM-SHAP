---
title: "estimation"
type: module
domain: estimation
parent: "[[mllm-shap]]"
tags: [architecture, estimation]
created: 2026-10-07
updated: 2026-10-18
related-code:
  - "[[src/mllm_shap/estimation/__init__.py]]"
  - "[[src/mllm_shap/estimation/_shared.py]]"
  - "[[src/mllm_shap/estimation/complementary.py]]"
  - "[[src/mllm_shap/estimation/loo.py]]"
  - "[[src/mllm_shap/estimation/kernel_banzhaf.py]]"
  - "[[src/mllm_shap/estimation/leverage.py]]"
  - "[[src/mllm_shap/estimation/proxy.py]]"
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
- Rescale attributions for presentation (normalisers) and make Shapley values efficient (`additive_efficiency`).

## Public surface

- `ExactEstimator` (`exact`): all `2^n` masks in bitmask order; requires the full table.
- `MonteCarloEstimator` (`mc`): the grand and empty coalitions, singletons and `(n-1)`-sets, then a uniformly drawn size and a uniform unseen coalition of it. Stratified SVARM ( [[Papers#Kolpaczki 2024 — SVARM|paper]], [[Estimation decisions#2026-10-18 — Stratified aggregation for mc|decision]]).
- `ComplementaryEstimator` (`complementary`): `(N, ∅)`, singleton pairs, block pairs per size `2..n/2` that put every player on both sides, then random pairs. `m_counts` exposes the count matrix. A warning names sizes that stay empty ([[Estimation decisions#2026-10-07 — Singleton-first complementary sampling|decision]]).
- `NeymanEstimator` (`neyman_cc`, default): phase one yields `(N, ∅)`, singleton pairs, then complete passes (one draw per player) size by size, ending between passes. `begin_allocation(masks, payoffs)` water-fills the remaining pairs over sizes `>= n/2` with weight `sqrt(sum sigma^2[:, j]/j + sum sigma^2[:, n-j]/(n-j))` (the middle column once, divided by `n`); `sample_allocation_masks()` yields phase two.
- `LeverageEstimator` (`leverage`, Shapley): complementary pairs allocated uniformly over sizes, enumerating covered size classes; `leverage_values` solves the sum-constrained regression with weight `(n-1)/(s(n-s) count_s)`, so values are efficient at every budget.
- `KernelBanzhafEstimator` (`kernel_banzhaf`, Banzhaf): distinct uniform complementary pairs; least squares of payoffs on presence minus one half plus an intercept.
- `ProxyEstimator` (`proxy`, both estimands, extra `proxy`): leverage-sampled masks and boosted trees (100, depth 3, seed 0) with exact per-leaf values (`tree_values`). The aggregator's values of the residual are added when `n < 30` or masks `>= 16n`. `scikit-learn` is imported at construction.
- `LeaveOneOutEstimator` (`loo`): `N` and each `N - {i}`; ignores budget and estimand ([[Estimation decisions#2026-10-12 — Leave-one-out bypasses the estimand|decision]]).
- Every estimator: `estimate_attributions(masks, payoffs, aggregator)`. `leverage` and `kernel_banzhaf` reject the other estimand; `complementary` and `neyman_cc` accept Shapley only.
- `ShapleyAggregator`, `BanzhafAggregator`: a complete Shapley table goes through `aggregate_shapley_exact`. Other tables use `aggregate_from_marginals`: per player and size stratum `l` the mean payoff of sampled coalitions with `i` of size `l+1` minus the mean of those without `i` of size `l`, combined with stratum probabilities renormalised over strata having both means. A player with none gets 0 and one warning.
- Normalisers: `IdentityNormalizer`, `AbsSumNormalizer`, `PowerShiftNormalizer`, `MinMaxNormalizer`; `additive_efficiency(values, grand, base)` shifts all values equally (`explanation.efficiency: additive`).
- `_shared.py`: `budget_fraction(n, fraction, evaluations)` turns an absolute count (`BudgetConfig.evaluations_for` resolves `per_player` to `ceil(alpha n)`) into `evaluations / 2^n`; `compute_mc_num_samples` (floor 2, or `n + 2`) and `compute_complementary_num_splits` (even, floor `min(2n + 2, 2^n)`); `random_present`.

## Data flow

- In: `PlayerSet`, the budget fraction, `include_minimal_masks`, `seed` from the [[pipeline]] config; later the collected masks and payoffs (both phases for Neyman).
- Out: an iterator of `CoalitionMask` and a `list[float]` of attributions.
- Imports [[domain]], [[protocols]], [[plugins]] (context type only), numpy; `proxy` also scikit-learn.

## Invariants

- On the full table every estimator is efficient for the Shapley estimand ([[Estimation decisions#2026-10-07 — Full coalition table|Full coalition table]]); `leverage` is efficient at every budget.
- Every sampler yields the grand and empty coalitions first; they count toward the budget.
- Complementary aggregation does not depend on mask order; at the floor the game `v(S) = |S|` gives 1 per player ([[Fixed issues|C14]]).
- `mc` is exact on the full table and nearly unbiased from about `32n` masks; `kernel_banzhaf` and `proxy` with correction are exact on the full table.
- `additive_efficiency` never increases the L2 error: it projects onto the hyperplane holding the exact values.

## Known issues

- Fixed: M01, M02, M03, C01, C03, C04, C05, C10, C12, C13, C14 ([[Fixed issues]]).

## Tests

- [[tests/estimation/test_axioms.py]], [[tests/estimation/test_games.py]] (reference games in `_games.py`)
- [[tests/estimation/test_exact.py]], [[tests/estimation/test_loo.py]], [[tests/estimation/test_monte_carlo.py]], [[tests/estimation/test_complementary.py]]
- [[tests/estimation/test_neyman.py]], [[tests/estimation/test_neyman_internals.py]]
- [[tests/estimation/test_leverage.py]], [[tests/estimation/test_kernel_banzhaf.py]], [[tests/estimation/test_proxy.py]]
- [[tests/estimation/test_shared.py]], [[tests/estimation/test_normalizers.py]]
- [[tests/estimation/estimands/test_weights.py]], [[tests/estimation/estimands/test_divergence.py]]
- [[tests/benchmarks/test_estimation_bench.py]]
