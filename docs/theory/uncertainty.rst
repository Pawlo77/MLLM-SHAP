Permutation, LIME, standard errors and budgets
==============================================

This page covers what :doc:`approximation` leaves out: two more estimators, the error a
sampled run reports about itself, when to stop sampling and how to turn a target error into a
budget.

.. _theory-permutation-lime:

Permutation sampling and the LIME surrogate
-------------------------------------------

``permutation`` draws a random ordering of the players and evaluates every prefix of it. With
``antithetic: true`` (the default) it also takes the reverse ordering. The reverse's prefix of
size :math:`k` is the complement of the first ordering's prefix of size :math:`n - k`.

The prefix of size :math:`k` of a uniform ordering is a uniform subset of size :math:`k`.
The prefixes therefore feed the same size-stratified aggregation as ``mc``, and any estimand
works. Prefixes of one ordering are correlated, which the standard errors below do not model.

``lime`` fits the surrogate of Ribeiro et al. (2016). Random subsets :math:`S` get the weight
:math:`w(S) = \sqrt{\exp(-d(S)^2 / \sigma^2)}` with :math:`\sigma = 25` and
:math:`d(S) = 100\,(1 - \sqrt{|S|/n})`, the cosine distance of the presence vector to the
all-present one times 100. The fit is a ridge regression with an unpenalised intercept:

.. math::

   \min_{b,\beta}\; \sum_S w(S)\,\bigl(v(S) - b - \beta^\top x_S\bigr)^2 + \alpha \lVert\beta\rVert^2

The coefficients are not Shapley values. They depend on the kernel and the penalty, need not
sum to :math:`v(N) - v(\emptyset)` and ignore the estimand.

.. _theory-standard-errors:

Standard errors
---------------

For ``mc`` the estimate of player :math:`i` is
:math:`\hat\phi_i = \sum_l q_l\,(\bar v^{+}_{i,l} - \bar v^{-}_{i,l})`. In size stratum :math:`l`,
:math:`\bar v^{+}_{i,l}` is the mean payoff of sampled coalitions of size :math:`l + 1` that
contain :math:`i`, and :math:`\bar v^{-}_{i,l}` that of sampled size-:math:`l` coalitions that
do not. The :math:`q_l` are the semivalue's stratum probabilities, renormalised over the strata
that have both means.

Coalitions of one size are a simple random sample, so each mean has variance
:math:`(s^2 / n_s)(1 - n_s / N_l)` with :math:`N_l = \binom{n-1}{l}`; the second factor is the
finite-population correction. The strata are independent:

.. math::

   \operatorname{Var}(\hat\phi_i) = \sum_l q_l^2 \bigl(\operatorname{Var}\bar v^{+}_{i,l} + \operatorname{Var}\bar v^{-}_{i,l}\bigr).

A side with one observation uses the pooled variance of all payoffs. An enumerated stratum has
no error, so a full table gives zero.

On the seven reference games of the test suite, the intervals
:math:`\hat\phi_i \pm 1.96\,\widehat{\mathrm{se}}_i` held the exact value in 93.3 to 95.0% of
2,000 seeds at 25% budget. The same formula covered ``permutation`` only 60 to 82% of the time.
Sandwich estimators for ``leverage`` and ``kernel_banzhaf`` were miscalibrated. Those estimators
report no standard error.

Stopping
--------

Every ``check_every`` coalitions past a floor, the rule asks two questions. Is the widest
interval at most ``max_ci_width``? Are the ``top_k_stable`` players of largest magnitude the same
set as at the previous check? Either answer ends sampling, and the budget stays the upper bound.

On the reference games the rule spent about what a fixed budget of the same error spends, with 0
to 7% fewer evaluations. Use it as an accuracy control, not as a large saving.

Planning a budget
-----------------

Suppose marginal contributions lie in a range of width :math:`r`. For a target error
:math:`\epsilon` at confidence :math:`1 - \beta`, Hoeffding's inequality with a union bound over
the :math:`n` players asks for

.. math::

   m \ge \frac{r^2 \ln(2n / \beta)}{2\,\epsilon^2}

random orderings (Maleki et al., 2013). Each ordering touches :math:`n - 1` new coalitions.
``mllm-shap plan`` evaluates the bound. It is a worst case and loose, so a pilot range gives a
tighter budget.

References
----------

- Maleki, S., Tran-Thanh, L., Hines, G., Rahwan, T., Rogers, A. (2013). Bounding the
  Estimation Error of Sampling-based Shapley Value Approximation. arXiv:1306.4265.
  https://arxiv.org/abs/1306.4265
- Ribeiro, M. T., Singh, S., Guestrin, C. (2016). "Why Should I Trust You?": Explaining the
  Predictions of Any Classifier. *KDD*. https://arxiv.org/abs/1602.04938

Further reading
---------------

- Sampling estimators: :doc:`approximation`
- Usage: :doc:`../guides/choosing_an_estimator`
- API: :doc:`../api/estimation`
