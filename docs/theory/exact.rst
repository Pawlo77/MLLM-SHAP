Exact estimation
================

The :class:`~nlp_shap.estimation.exact.ExactEstimator` lazily enumerates every
coalition, from the empty coalition to the grand coalition, and delegates
attribution to an estimand plugin. Use it as ground truth on tiny player sets
before trusting approximate samplers.

Enumeration scope
-----------------

For :math:`n` players the estimator evaluates every coalition:

.. math::

   |\mathcal{S}_{\mathrm{exact}}| = 2^n

Inside :class:`~nlp_shap.ExplainRunner` the grand coalition :math:`v(N)` is scored
from the base generation (the output for the unmasked prompt), so it is served
from the result store instead of a second backend call. With :math:`v(\emptyset)` and
:math:`v(N)` both present, Shapley values satisfy efficiency:
:math:`\sum_i \phi_i = v(N) - v(\emptyset)`. The run output exposes both
payoffs as :attr:`~nlp_shap.pipeline.result.ExplainResult.base_value` and
:attr:`~nlp_shap.pipeline.result.ExplainResult.grand_value`.

Complexity grows exponentially in :math:`n`; reserve exact estimation for toy
models, sanity checks, and estimator validation.

Estimand delegation
-------------------

Exact enumeration produces coalition payoffs :math:`v(S)`. The estimand plugin
(Shapley or Banzhaf) applies coalition weights and marginal contributions:

.. math::

   \text{ExactEstimator} \rightarrow \{v(S)\}_{S}
   \xrightarrow{\text{EstimandAggregator}}
   (\phi_1, \ldots, \phi_n)

Monte Carlo and Neyman estimators reuse the same estimand boundary — exact
estimation establishes the reference path for validating approximate samplers.

Legacy parity
-------------

The implementation matches the Shapley values of ``PreciseShapExplainer`` from
`MLLM-Shap <https://github.com/Pawlo77/MLLM-Shap>`_ on additive characteristic
functions; both aggregate the full :math:`2^n` table. See
:doc:`../guides/migrating_from_mllm_shap` for the other differences.

Further reading
---------------

- Estimand definitions: :doc:`estimands`
- Cooperative-game foundations: :doc:`cooperative_games`
- Usage: :doc:`../guides/exact`
- API: :class:`~nlp_shap.estimation.exact.ExactEstimator`
