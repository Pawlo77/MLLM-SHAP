Approximate estimation
======================

When :math:`n` players make exact enumeration infeasible, ``mllm-shap`` approximates
attributions from a budget of evaluated coalitions. Three samplers aggregate by
averaging: :class:`~mllm_shap.estimation.monte_carlo.MonteCarloEstimator`
(Shapley or Banzhaf), :class:`~mllm_shap.estimation.complementary.ComplementaryEstimator`
(symmetric pairs :math:`(S, N\setminus S)`, complementary-contribution aggregation)
and :class:`~mllm_shap.estimation.neyman.NeymanEstimator` (two-phase
complementary sampling with Neyman allocation). Three fit a regression:
:doc:`regression_estimators`.

Budget model
------------

``budget`` takes exactly one of three keys; all count the empty and grand
coalitions:

.. list-table::
   :header-rows: 1

   * - Key
     - Coalitions evaluated
   * - ``fraction`` in :math:`(0, 1]`
     - :math:`\lfloor f\,2^n \rfloor`
   * - ``evaluations``
     - the given count, at most :math:`2^n`
   * - ``per_player`` :math:`\alpha`
     - :math:`\lceil \alpha n \rceil`, as ``evaluations``

A fraction of :math:`2^n` is meaningless beyond about 20 players, so the
literature budgets in multiples of :math:`n`. Each estimator raises the budget to
its floor (2 for ``mc``, :math:`2n + 2` for the paired estimators, which also round
down to whole pairs). Complementary and Neyman sampling yield :math:`(N, \emptyset)`
first, then the :math:`n` singleton pairs :math:`(\{i\}, N\setminus\{i\})`.
:math:`v(N)` is scored from the base generation, so it costs no extra call.

Monte Carlo and estimand delegation
-----------------------------------

Monte Carlo evaluates the empty, grand, singleton and :math:`(n-1)`-player
coalitions, then draws a size uniformly among the sizes with unseen coalitions and
a uniform coalition of that size, without repeats (Stratified SVARM). It does
**not** embed Shapley weights: the selected estimand plugin aggregates the payoffs,
so the same masks give different values under Shapley and Banzhaf.

.. _theory-partial-samples:

What the aggregator computes from partial samples
-------------------------------------------------

On a complete table the estimand plugins return the exact value. On a partial
sample every coalition updates every player. Let :math:`\bar v^{+}_{i,\ell}` be the
mean payoff of the sampled coalitions that contain :math:`i` and have size
:math:`\ell + 1`, and :math:`\bar v^{-}_{i,\ell}` that of the sampled coalitions
that exclude :math:`i` and have size :math:`\ell`. With :math:`L_i` the strata
having both means and :math:`p_\ell` the semivalue's stratum probability
(:math:`1/n` for Shapley, :math:`\binom{n-1}{\ell} / 2^{n-1}` for Banzhaf),

.. math::

   \hat\phi_i = \frac{\sum_{\ell \in L_i} p_\ell\,
   (\bar v^{+}_{i,\ell} - \bar v^{-}_{i,\ell})}{\sum_{\ell \in L_i} p_\ell}

A player with no such stratum gets 0 and a warning. The estimate is consistent
but not unbiased at low coverage: dropped strata make the player inherit the
mean difference of the observed ones. Efficiency,
:math:`\sum_i \hat\phi_i = v(N) - v(\emptyset)`, holds exactly only on the full
table, for ``leverage``, and with ``explanation.efficiency: additive``.

Complementary contributions
---------------------------

Complementary sampling evaluates symmetric pairs and accumulates CC statistics in
:math:`M_{i,j}` (how often player :math:`i` appears in size-:math:`j` coalitions)
and :math:`C_{i,j}` (summed complementary contributions). For pair
:math:`(S, N\setminus S)` with payoffs :math:`v(S)` and :math:`v(N\setminus S)`:

.. math::

   u = v(S) - v(N\setminus S)

Each :math:`u` updates :math:`C` for players present in :math:`S` and
:math:`N\setminus S`. Final attributions combine element-wise ratios
:math:`C_{i,j}/M_{i,j}` across coalition sizes:

.. math::

   \hat\phi_i = \frac{1}{n} \sum_{j=1}^{n} \frac{C_{i,j}}{M_{i,j}}

The column :math:`j = n` comes only from the pair :math:`(N, \emptyset)`, which
is sampled exactly once and contributes :math:`(v(N) - v(\emptyset))/n` to every
player. Without it every player would lose its :math:`S = N\setminus\{i\}`
marginal. Masks are matched to their complements by value, so the aggregation
does not depend on the order in which coalitions were evaluated.

A cell with :math:`M_{i,j} = 0` contributes zero to :math:`\hat\phi_i`, which
biases partial budgets. The sampler therefore yields the singleton pairs after
:math:`(N, \emptyset)`, then for each size :math:`2 \le \ell \le n/2` pairs over
blocks of :math:`\ell` players that partition the players, so every player is on
both sides of some pair of that size. That costs about :math:`n \ln n` pairs;
below it, random pairs follow and a warning names the sizes that stay empty.

Neyman allocation
-----------------

Neyman-CC extends complementary sampling with two phases:

1. **Initial sampling** — after the boundary and singleton pairs, structured
   draws fill every :math:`M_{i,j}` cell up to :math:`m_{\text{init}}` times,
   capped by the :math:`\binom{n-1}{j-1}` size-:math:`j` coalitions that contain
   player :math:`i`. A pass draws once per player at one size, from the size-:math:`j`
   coalitions that contain that player, so the draws stay uniform within a size.
   Passes run size by size and the phase ends between passes: when every feasible
   cell is filled, a pass adds nothing, or the budget cannot fit another pass.
2. **Neyman allocation** — the remaining budget is split across coalition sizes
   :math:`j \ge \lceil n/2 \rceil` proportional to estimated standard deviations
   :math:`\hat\sigma_{i,j}`, never assigning a size more pairs than it has left
   unsampled. When every variance estimate is zero the remaining pairs are split
   by the number of unsampled pairs per size.

Variance estimates use the complementary sample variance:

.. math::

   \hat\sigma^2_{i,j} = \frac{1}{M_{i,j}-1}
   \left(\sum u^2 - \frac{(\sum u)^2}{M_{i,j}}\right)

Allocation follows the Neyman rule (symmetric halves summed):

.. math::

   \hat M_j \propto
   \sqrt{
     \sum_{i} \frac{\hat\sigma^2_{i,j}}{j}
     + \sum_{i} \frac{\hat\sigma^2_{i,n-j}}{n-j}
   }

where :math:`n - j` is the size of the complement. Derivation: a pair with larger
side :math:`j` puts player :math:`i` in a size-:math:`j` coalition with probability
:math:`j/n`, so cell :math:`(i, j)` is observed :math:`m_j j/n` times and cell
:math:`(i, n-j)` :math:`m_j (n-j)/n` times. The variance
:math:`\sum \hat\sigma^2_{i,c} / M_{i,c}` is minimised under
:math:`\sum_j m_j = m` by the rule above. For :math:`j = n/2` both sides are one
column and count once, as :math:`\sum_i \hat\sigma^2_{i,j} / n`.

**Small budgets:** phase one may consume the whole allowance before allocation
begins, and then behaves like complementary sampling.

Because allocation depends on phase-one payoffs, Neyman requires a two-step API:
:meth:`~mllm_shap.estimation.neyman.NeymanEstimator.sample_masks` (phase one),
:meth:`~mllm_shap.estimation.neyman.NeymanEstimator.begin_allocation`, then
:meth:`~mllm_shap.estimation.neyman.NeymanEstimator.sample_allocation_masks`.
:class:`~mllm_shap.ExplainRunner` runs both phases: any estimator that implements
:class:`~mllm_shap.protocols.estimator.TwoPhaseEstimator` gets its phase-one
payoffs and a second scheduling pass for the allocated masks.

.. _theory-choosing-a-budget:

Choosing a budget
-----------------

Convergence depends on the game. :doc:`../guides/estimator_benchmark` tabulates the
error of every estimator at budgets of :math:`2n` to :math:`32n` coalitions on five
reference games; run the same comparison on a game that resembles your prompts, or
on the prompt itself with a cheap backend and ``exact`` as the reference.

References
----------

- Zhang, J., Sun, Q., Liu, J., Xiong, L., Pei, J., Ren, K. (2023). Efficient
  Sampling Approaches to Shapley Value Approximation. *Proc. ACM Manag. Data*
  1(1). https://doi.org/10.1145/3588728
- Neyman, J. (1934). On the Two Different Aspects of the Representative Method.
  *J. R. Stat. Soc.* 97(4). https://doi.org/10.2307/2342192
- Castro, J., Gómez, D., Tejada, J. (2009). Polynomial calculation of the Shapley
  value based on sampling. *Comput. Oper. Res.* 36(5).
  https://doi.org/10.1016/j.cor.2008.04.004
- Mitchell, R., Cooper, J., Frank, E., Holmes, G. (2022). Sampling Permutations
  for Shapley Value Estimation. *Journal of Machine Learning Research* 23(43).
- Goldshmidt, R., Horovicz, M. (2024). TokenSHAP: Interpreting Large Language
  Models with Monte Carlo Shapley Value Estimation. arXiv:2407.10114.
  https://arxiv.org/abs/2407.10114

Further reading
---------------

- Exact reference path: :doc:`exact`
- Estimand definitions: :doc:`estimands`
- Usage: :doc:`../guides/approximation`
- API: :doc:`../api/estimation`
