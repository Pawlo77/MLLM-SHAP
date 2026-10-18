---
title: "Papers"
type: paper-list
domain: theory
parent: "[[mllm-shap]]"
created: 2026-10-07
updated: 2026-10-08
tags: [papers]
related-code:
  - "[[src/mllm_shap/estimation/estimands/shapley.py]]"
  - "[[src/mllm_shap/estimation/estimands/banzhaf.py]]"
  - "[[src/mllm_shap/estimation/estimands/_weights.py]]"
  - "[[src/mllm_shap/estimation/monte_carlo.py]]"
  - "[[src/mllm_shap/estimation/complementary.py]]"
  - "[[src/mllm_shap/estimation/neyman.py]]"
  - "[[src/mllm_shap/estimation/_shared.py]]"
  - "[[src/mllm_shap/masking/partitions.py]]"
  - "[[src/mllm_shap/value/tfidf.py]]"
---
# Papers

Papers the library and its theory pages cite, by first author; digital object identifier (DOI) or arXiv links only, never PDFs.

## Banzhaf 1965 — Weighted voting doesn't work

Citation: John F. Banzhaf III, "Weighted voting doesn't work: A mathematical analysis", *Rutgers Law Review* 19(2), 1965; no DOI or arXiv record.

Claim: The `banzhaf` estimand is the unnormalised index extended to any payoff, the plain average of marginal contributions with weight `1/2^(n-1)`; the normalised index is not implemented.

Used by: [[estimation]], `docs/theory/estimands.rst`.

## Castro 2009 — Polynomial calculation of the Shapley value

Citation: Javier Castro, Daniel Gómez, Juan Tejada, "Polynomial calculation of the Shapley value based on sampling", *Computers & Operations Research* 36(5), 2009, [doi:10.1016/j.cor.2008.04.004](https://doi.org/10.1016/j.cor.2008.04.004).

Claim: Partial-sample estimates must be consistent, converging to the exact value as samples grow; mllm-shap samples coalitions, not the paper's player orderings.

Used by: [[Estimation decisions#2026-10-18 — Stratified aggregation for mc|stratified aggregation]], `docs/theory/approximation.rst`.

## Dubey 1979 — Mathematical properties of the Banzhaf power index

Citation: Pradeep Dubey, Lloyd S. Shapley, "Mathematical properties of the Banzhaf power index", *Mathematics of Operations Research* 4(2), 1979, [doi:10.1287/moor.4.2.99](https://doi.org/10.1287/moor.4.2.99).

Claim: Banzhaf is the expected marginal contribution when every other player joins with probability one half and is not efficient, so mllm-shap keeps it a separate estimand never compared with Shapley as the same quantity.

Used by: [[estimation]], `docs/theory/estimands.rst`.

## Ethayarajh 2021 — Attention flows are Shapley value explanations

Citation: Kawin Ethayarajh, Dan Jurafsky, "Attention flows are Shapley value explanations", *ACL-IJCNLP 2021 (short papers)*, [arXiv:2105.14652](https://arxiv.org/abs/2105.14652).

Claim: Attention flow is a Shapley value of a max-flow game while raw attention is not, which supports prompt tokens as players and output-based value functions; mllm-shap reads no model internals.

Used by: [[domain]], `docs/theory/estimands.rst`.

## Fryer 2021 — Shapley values for feature selection

Citation: Daniel Fryer, Inga Strümke, Hien Nguyen, "Shapley values for feature selection: The good, the bad, and the axioms", *IEEE Access* 9, 2021, [arXiv:2102.10936](https://arxiv.org/abs/2102.10936).

Claim: Shapley axioms can mislead feature selection, the argument for offering several estimands and labelling every result with its estimand.

Used by: [[estimation]], `docs/theory/estimands.rst`, `docs/guides/applications.rst`.

## Goldshmidt 2024 — TokenSHAP

Citation: Roni Goldshmidt, Miriam Horovicz, "TokenSHAP: Interpreting large language models with Monte Carlo Shapley value estimation", arXiv preprint, 2024, [arXiv:2407.10114](https://arxiv.org/abs/2407.10114).

Claim: The default game: whitespace tokens as players, deletion as absence and response similarity to the base generation (`tfidf_cosine`) as payoff; its Monte Carlo sampling scheme is not used.

Used by: [[value]], [[Base generation]], `docs/index.rst`, `docs/theory/approximation.rst`.

## Lundberg 2017 — A unified approach to interpreting model predictions

Citation: Scott M. Lundberg, Su-In Lee, "A unified approach to interpreting model predictions", *NeurIPS 2017*, [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).

Claim: Attribution is a cooperative game over input features whose empty-coalition output is the base value efficiency is measured against; Kernel SHapley Additive exPlanations (SHAP) regression and model-specific explainers are not implemented.

Used by: [[estimation]], `docs/theory/shapley_values.rst`, `docs/theory/estimands.rst`, `docs/guides/applications.rst`.

## Mitchell 2022 — Sampling permutations for Shapley value estimation

Citation: Rory Mitchell, Joshua Cooper, Eibe Frank, Geoffrey Holmes, "Sampling permutations for Shapley value estimation", *Journal of Machine Learning Research* 23, 2022, [arXiv:2104.12199](https://arxiv.org/abs/2104.12199).

Claim: Estimators are compared at equal evaluation budgets; its antithetic, herding and quasi-Monte Carlo permutation sets are not implemented.

Used by: [[estimation]], [[Estimation decisions#2026-10-18 — Stratified aggregation for mc|stratified aggregation]], `docs/theory/approximation.rst`.

## Neyman 1934 — Two aspects of the representative method

Citation: Jerzy Neyman, "On the two different aspects of the representative method: the method of stratified sampling and the method of purposive selection", *Journal of the Royal Statistical Society* 97(4), 1934, [doi:10.2307/2342192](https://doi.org/10.2307/2342192).

Claim: Phase two of `neyman_cc` allocates the remaining budget across coalition sizes in proportion to the estimated standard deviation of their complementary contributions, the optimal allocation `n_h ∝ N_h S_h`.

Used by: [[estimation]], `docs/theory/approximation.rst`.

## Owen 1972 — Multilinear extensions of games

Citation: Guillermo Owen, "Multilinear extensions of games", *Management Science* 18(5), 1972, [doi:10.1287/mnsc.18.5.64](https://doi.org/10.1287/mnsc.18.5.64).

Claim: Shapley and Banzhaf are two weightings of the same marginal contributions (the diagonal integral and the midpoint of the multilinear extension), as the theory pages present them; diagonal sampling is not implemented.

Used by: [[estimation]], `docs/theory/cooperative_games.rst`.

## Shapley 1953 — A value for n-person games

Citation: Lloyd S. Shapley, "A value for n-person games", *Contributions to the Theory of Games II*, 1953, [doi:10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018).

Claim: The `shapley` estimand, its weights `shapley_weight(k, n)` and exact aggregation, with efficiency (values sum to `v(N) - v(∅)`) as the check every estimator must pass.

Used by: [[estimation]], `docs/theory/cooperative_games.rst`, `docs/theory/shapley_values.rst`, `docs/theory/estimands.rst`.

## Strumbelj 2014 — Explaining prediction models with feature contributions

Citation: Erik Štrumbelj, Igor Kononenko, "Explaining prediction models and individual predictions with feature contributions", *Knowledge and Information Systems* 41(3), 2014, [doi:10.1007/s10115-013-0679-x](https://doi.org/10.1007/s10115-013-0679-x); the theory pages cited a wrong DOI until 0.1.18 ([[Fixed issues|N01]]).

Claim: Sampling is the way past the `2^n` table, behind the `mc` estimator, which samples coalitions uniformly rather than orderings with reference instances.

Used by: [[estimation]], `docs/theory/estimands.rst`.

## Young 1985 — Monotonic solutions of cooperative games

Citation: H. Peyton Young, "Monotonic solutions of cooperative games", *International Journal of Game Theory* 14(2), 1985, [doi:10.1007/BF01769885](https://doi.org/10.1007/BF01769885).

Claim: Efficiency, symmetry and strong monotonicity single out the Shapley value, so a token whose presence helps more in every context gets more credit.

Used by: [[estimation]], `docs/theory/shapley_values.rst`.

## Zhang 2023 — Efficient sampling approaches to Shapley value approximation

Citation: Jiayao Zhang, Qiheng Sun, Jinfei Liu, Li Xiong, Jian Pei, Kui Ren, "Efficient sampling approaches to Shapley value approximation", *Proceedings of the ACM on Management of Data* 1(1), 2023, [doi:10.1145/3588728](https://doi.org/10.1145/3588728).

Claim: The `complementary` estimator (count matrix `M` and summed contributions `C` per player and size) and the two-phase `neyman_cc`; the size-`n` stratum is informed only by `(N, ∅)` ([[Fixed issues|C03]]), and the empirical Bernstein allocation is not implemented.

Used by: [[estimation]], [[Estimation decisions#2026-10-07 — Empty and grand pair in CC sampling|empty and grand pair]], `docs/theory/approximation.rst`.

## Kolpaczki 2024 — SVARM

Citation: Patrick Kolpaczki, Viktor Bengs, Maximilian Muschalik, Eyke Hüllermeier, "Approximating the Shapley Value without Marginal Contributions", *AAAI* 38(12), 2024, [doi:10.1609/aaai.v38i12.29225](https://doi.org/10.1609/aaai.v38i12.29225).

Claim: `mc` takes the size-stratified with-and-without means, so one evaluation updates every player, and the exact evaluation of the smallest strata first; the variance guarantees are not used.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.

## Liu 2024 — Kernel Banzhaf

Citation: Yurong Liu, R. Teal Witter, Flip Korn, Tarfah Alrashed, Dimitris Paparas, Christopher Musco, Juliana Freire, "Kernel Banzhaf: A Fast and Robust Estimator for Banzhaf Values", 2024, [arXiv:2410.08336](https://arxiv.org/abs/2410.08336).

Claim: `kernel_banzhaf` fits a linear model to payoffs of paired uniform coalitions; its slopes are the Banzhaf values.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.

## Liu 2025 — AttriBoT

Citation: Fengyuan Liu, Nikhil Kandpal, Colin Raffel, "AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution", *ICLR*, 2025, [arXiv:2411.15102](https://arxiv.org/abs/2411.15102).

Claim: Hierarchical runs rely on its observation that paragraph scores approximate summed sentence scores; the other tricks are not implemented.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.

## Musco 2025 — Leverage SHAP

Citation: Christopher Musco, R. Teal Witter, "Provably Accurate Shapley Value Estimation via Leverage Score Sampling", *ICLR*, 2025, [arXiv:2410.01917](https://arxiv.org/abs/2410.01917).

Claim: `leverage` samples coalition sizes by leverage score (uniform over sizes) with paired sampling and solves the constrained Kernel SHAP regression; the authors' without-replacement Bernoulli scheme is replaced by deterministic per-size counts.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.

## Thies 2026 — ProxySHAP

Citation: Santo M. A. R. Thies, Hubert Baniecki, R. Teal Witter, Eyke Hüllermeier, Maximilian Muschalik, Fabian Fumagalli, "ProxySHAP", *NeurIPS*, 2026, [arXiv:2605.22738](https://arxiv.org/abs/2605.22738).

Claim: `proxy` fits a tree proxy and extracts its exact Shapley and Banzhaf values; only order 1 is implemented.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.

## Wang 2023 — Data Banzhaf

Citation: Jiachen T. Wang, Ruoxi Jia, "Data Banzhaf: A Robust Data Valuation Framework for Machine Learning", *AISTATS*, PMLR 206, 2023, [arXiv:2205.15466](https://arxiv.org/abs/2205.15466).

Claim: The Banzhaf value is the slope of the best linear fit under uniform coalitions and tolerates noisy payoffs, which motivates `kernel_banzhaf`.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.

## Witter 2025 — Regression-adjusted Monte Carlo

Citation: R. Teal Witter, Yurong Liu, Christopher Musco, "Regression-Adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values", *NeurIPS*, 2025, [arXiv:2506.11849](https://arxiv.org/abs/2506.11849).

Claim: `proxy` adds the sample-reuse estimate of the residual to the proxy's exact values.

Used by: [[estimation]], `docs/theory/regression_estimators.rst`.
