Regression estimators and hierarchical players
==============================================

``leverage``, ``kernel_banzhaf`` and ``proxy`` fit a model to the archived masks and
payoffs instead of averaging marginals, so every evaluated coalition informs every
player and ``reanalyze`` can recompute them from the archive alone. All three keep
the empty and grand coalitions, which count toward the budget, and draw
complementary pairs :math:`(S, N\setminus S)`; the budget floor is :math:`2n + 2`
masks. :doc:`../guides/estimator_benchmark` compares them with the samplers.

Leverage SHAP
-------------

Shapley values solve the constrained regression

.. math::

   \hat\phi = \arg\min_{\sum_i \phi_i = v(N) - v(\emptyset)}
   \sum_{0 < |S| < n} k(S)\,\big(v(S) - v(\emptyset) - \mathbf 1_S^\top \phi\big)^2,
   \qquad k(S) = \frac{n - 1}{\binom{n}{|S|}\,|S|\,(n - |S|)} .

The leverage score of a coalition of size :math:`s` is :math:`\binom{n}{s}^{-1}`
(Musco and Witter 2025), so every size carries the same total leverage and sizes
are sampled uniformly. ``leverage`` allocates pairs to size classes in proportion
to that mass, enumerates a class whose share covers all its pairs, and samples
the others without replacement. A sampled size :math:`s` with :math:`m_s`
coalitions gets the weight :math:`k(S)\binom{n}{s} / m_s` (:math:`k(S)` for an
enumerated size), computed from the masks alone. The constraint is solved by
eliminating one player, so the values add up to :math:`v(N) - v(\emptyset)` at
every budget and equal the exact values on the full table. Shapley only.

Kernel Banzhaf
--------------

The Banzhaf value of player :math:`i` is the slope of the best linear fit of
:math:`v` on the presence indicators under the uniform distribution over
coalitions (Kernel Banzhaf, Liu et al. 2024; Data Banzhaf, Wang and Jia 2023).
``kernel_banzhaf`` draws distinct uniform pairs
and fits :math:`v(S) \approx \beta_0 + \sum_i (\mathbf 1_{i \in S} - \tfrac12)\beta_i`
by least squares. The centred columns are orthogonal on the full table, so the
fit is exact there. Banzhaf only.

ProxySHAP
---------

``proxy`` (extra ``mllm-shap[proxy]``, scikit-learn) follows ProxySHAP of order 1
(Thies et al. 2026). It fits gradient-boosted trees (100 trees, depth 3, fixed
seed) to the leverage-sampled masks. A leaf is the game
:math:`c\,[P \subseteq S,\ A \cap S = \emptyset]` for the players :math:`P` present
and :math:`A` absent on its path, with :math:`p = |P|`, :math:`a = |A|`. Its
Shapley values are :math:`c\,(p-1)!\,a!/(p+a)!` for players in :math:`P` and
:math:`-c\,p!\,(a-1)!/(p+a)!` for players in :math:`A`; its Banzhaf values are
:math:`\pm c / 2^{p+a-1}`. Summing over leaves gives the exact values of the
proxy. With fewer than 30 players, or at least 16 masks per player, the values of
the residual :math:`v - \text{proxy}` on the sampled coalitions, by stratified
means, are added back (regression-adjusted Monte Carlo, Witter et al. 2025), so
the estimate is exact on the full table. Both estimands.

Monte Carlo and the efficiency option
-------------------------------------

``mc`` evaluates the empty, grand, singleton and :math:`(n-1)`-player coalitions,
then draws a size uniformly from the sizes with unseen coalitions and a uniform
coalition of that size (Stratified SVARM, Kolpaczki et al. 2024). With
``explanation.efficiency: additive`` the Shapley values are shifted equally so
they sum to :math:`v(N) - v(\emptyset)`. The shift is the projection onto the
efficiency hyperplane, which contains the exact values, so it never increases
the :math:`\ell_2` error.

Hierarchical players
--------------------

For long prompts, ``players: sentences`` makes one player per sentence and
:meth:`~mllm_shap.ExplainRunner.explain_hierarchical` runs two games with the
same estimator and budget rule. The coarse game has one player per paragraph (or
message, ``hierarchical.group_by``). The ``keep_fraction`` of groups with the
largest absolute values is refined: their sentences are the players of the second
game and all other text stays present. Paragraph scores approximate the sums of
their sentence scores (Liu et al. 2025), so the top sentences usually lie in
the kept groups, at a fraction of the calls of the flat game. The fine values are
Shapley (or Banzhaf) values of that second game, not of the flat one. Both
games are archived under their own run ids.

References
----------

- Musco, C., Witter, R. T. (2025). Provably Accurate Shapley Value Estimation via
  Leverage Score Sampling. *ICLR 2025*. https://arxiv.org/abs/2410.01917
- Liu, Y., Witter, R. T., Korn, F., Alrashed, T., Paparas, D., Musco, C., Freire, J.
  (2024). Kernel Banzhaf: A Fast and Robust Estimator for Banzhaf Values.
  arXiv:2410.08336. https://arxiv.org/abs/2410.08336
- Wang, J. T., Jia, R. (2023). Data Banzhaf: A Robust Data Valuation Framework for
  Machine Learning. *AISTATS 2023*, PMLR 206. https://proceedings.mlr.press/v206/wang23e.html
- Thies, S. M. A. R., Baniecki, H., Witter, R. T., Hüllermeier, E., Muschalik, M.,
  Fumagalli, F. (2026). ProxySHAP. *NeurIPS 2026*. https://arxiv.org/abs/2605.22738
- Witter, R. T., Liu, Y., Musco, C. (2025). Regression-Adjusted Monte Carlo
  Estimators for Shapley Values and Probabilistic Values. *NeurIPS 2025*.
  https://arxiv.org/abs/2506.11849
- Kolpaczki, P., Bengs, V., Muschalik, M., Hüllermeier, E. (2024). Approximating the
  Shapley Value without Marginal Contributions. *AAAI 2024* 38(12).
  https://doi.org/10.1609/aaai.v38i12.29225
- Liu, F., Kandpal, N., Raffel, C. (2025). AttriBoT: A Bag of Tricks for Efficiently
  Approximating Leave-One-Out Context Attribution. *ICLR 2025*.
  https://arxiv.org/abs/2411.15102
