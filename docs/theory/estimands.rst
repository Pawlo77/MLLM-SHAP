Estimands: Shapley vs Banzhaf
==============================

An **estimand** is the quantity your attribution procedure targets. In
``mllm-shap``, estimands are explicit types — not implicit behaviour of an
estimator. Monte Carlo coalition averaging does **not** automatically estimate
Shapley values; unweighted means over sampled coalitions estimate the **Banzhaf
index** unless Shapley weights are applied.

For cooperative-game foundations, see :doc:`cooperative_games`.

Shapley value (estimand)
------------------------

The Shapley value :math:`\phi_i` (definition, axioms and uniqueness:
:doc:`shapley_values`) weights each coalition of size :math:`k` by

.. math::

   w_{\mathrm{Shapley}}(k, n) = \frac{k!\,(n-k-1)!}{n!}

Implemented in :class:`~mllm_shap.estimation.estimands.shapley.ShapleyAggregator`.

Banzhaf index (estimand)
------------------------

The **Banzhaf index** (also called the Banzhaf–Coleman power index in voting
theory) uses **uniform** weights over coalitions :math:`S \subseteq N \setminus
\{i\}`:

.. math::

   \beta_i(v) = \frac{1}{2^{|N|-1}}
   \sum_{S \subseteq N \setminus \{i\}}
   \bigl(v(S \cup \{i\}) - v(S)\bigr)

Equivalently, :math:`w_{\mathrm{Banzhaf}}(k, n) = 2^{-(n-1)}` for every coalition
size :math:`k`.

Implemented in :class:`~mllm_shap.estimation.estimands.banzhaf.BanzhafAggregator`.

**Interpretation:** :math:`\beta_i` measures the average marginal contribution
across all coalitions, or — in threshold games — the probability that player
:math:`i` is **pivotal** when each player independently joins with probability
:math:`1/2`. Banzhaf indices need not satisfy **efficiency**:
:math:`\sum_i \beta_i(v) \neq v(N)` in general.

Banzhaf (1965) introduced the index in weighted-voting analysis; Dubey & Shapley
(1979) developed its mathematical properties on the space of characteristic
functions.

When Shapley and Banzhaf diverge
--------------------------------

For **additive** games :math:`v(S) = \sum_{i \in S} a_i`, both indices recover
:math:`a_i`.

For **non-additive** games they can disagree substantially. A standard example is
the **majority/threshold** game with :math:`|N|=3` and :math:`v(S)=1` iff
:math:`|S| \ge 2`, else :math:`0`:

+------------------+----------------------------------+
| Estimand         | Per-player value                 |
+==================+==================================+
| Shapley          | :math:`2/6 \approx 0.333`        |
+------------------+----------------------------------+
| Banzhaf          | :math:`0.5`                      |
+------------------+----------------------------------+

The test suite locks this divergence on identical coalition samples.

Choosing an estimand in practice
--------------------------------

**Prefer Shapley when:**

- Attributions must sum to the total utility gain (**efficiency**).
- You need the classical fairness axioms for stakeholder review.
- Reporting aligns with SHAP literature and local accuracy guarantees.

**Prefer Banzhaf when:**

- You care about *swing* / pivotal influence per token.
- You want uniform coalition averaging without factorial weights.
- You explicitly study voting-power-style sensitivity.

Regardless of choice, label outputs honestly. ``mllm-shap`` separates estimands
from estimators so archives and papers never misreport Banzhaf output as Shapley.

Monte Carlo and estimand confusion
----------------------------------

A common pitfall in explainability codebases is to:

1. Sample coalitions :math:`S_1, \dots, S_m` with an estimator
2. Average marginal contributions **without** Shapley weights
3. Label the result "Shapley"

Step 2 estimates a **Banzhaf-style** quantity (uniform coalition average), not
:math:`\phi`. Štrumbelj & Kononenko (2014) discuss coalition-based feature
contributions; Fryer et al. (2021) analyze when Shapley axioms justify feature
selection in ML. Always apply the estimand aggregator that matches the quantity
you intend to report.

Estimands under partial samples
-------------------------------

With a budget below the full table the aggregators see only some coalitions. The
Shapley aggregator then takes, per coalition size, the mean payoff of the sampled
coalitions with a player minus that of those without it and weights the sizes
equally (a stratified mean); the Banzhaf aggregator uses the same strata weighted
by their binomial probabilities :math:`\binom{n-1}{k} / 2^{n-1}`, renormalized
over the sizes that have both means.
Both estimates converge to their estimand as coverage grows, but only the full
table guarantees efficiency. See
:ref:`theory-partial-samples` for the formulas and
:doc:`../guides/choosing_an_estimator` for budgets.

Labelling in ``mllm-shap``
--------------------------

Every :class:`~mllm_shap.pipeline.result.ExplainResult` carries an
:class:`~mllm_shap.domain.estimands.Estimand` label. Run archives persist the
same label in :class:`~mllm_shap.RunManifest` so
downstream analysis and compliance reviews can verify which estimand was used.

The :class:`~mllm_shap.protocols.estimand.EstimandAggregator` protocol defines the
aggregation contract; concrete implementations live in
:mod:`mllm_shap.estimation.estimands`.

References
----------

- Shapley, L. S. (1953). A value for *n*-person games. In H. W. Kuhn & A. W.
  Tucker (Eds.), *Contributions to the Theory of Games II* (pp. 307–317).
  Princeton University Press.
  `DOI:10.1515/9781400881970-018 <https://doi.org/10.1515/9781400881970-018>`__
- Banzhaf, J. F. (1965). Weighted voting doesn't work: A mathematical analysis.
  *Rutgers Law Review*, 19(2), 317–343.
- Dubey, P., & Shapley, L. S. (1979). Mathematical properties of the Banzhaf
  power index. *Mathematics of Operations Research*, 4(2), 99–131.
  `DOI:10.1287/moor.4.2.99 <https://doi.org/10.1287/moor.4.2.99>`__
- Štrumbelj, E., & Kononenko, I. (2014). Explaining prediction models and
  individual predictions with feature contributions.
  *Knowledge and Information Systems*, 41(3), 647–665.
  `DOI:10.1007/s10115-013-0679-x <https://doi.org/10.1007/s10115-013-0679-x>`__
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *Advances in Neural Information Processing Systems*, 30.
  `ML Anthology <https://mlanthology.org/neurips/2017/lundberg2017neurips-unified/>`__;
  `arXiv:1705.07874 <https://arxiv.org/abs/1705.07874>`__
- Fryer, D. V., Strümke, I., & Nguyen, H. D. (2021). Shapley values for feature
  selection: The good, the bad, and the axioms. *IEEE Access*, 9, 144352–144360.
  `arXiv:2102.10936 <https://arxiv.org/abs/2102.10936>`__
- Ethayarajh, K., & Jurafsky, D. (2021). Attention flows are Shapley value
  explanations. In *Proceedings of the 59th Annual Meeting of the Association for
  Computational Linguistics and the 11th International Joint Conference on Natural
  Language Processing (Volume 2: Short Papers)* (pp. 49–54).
  `ACL Anthology <https://aclanthology.org/2021.acl-short.8/>`__;
  `arXiv:2105.14652 <https://arxiv.org/abs/2105.14652>`__
