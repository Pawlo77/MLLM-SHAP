Choosing an estimator
=====================

Every coalition an estimator asks for is one model generation (minus the ones
served from the cache), so the estimator and ``budget.fraction`` decide what a run
costs. This page turns the trade-offs into a decision table. :doc:`approximation`
has the code for the samplers, :doc:`../theory/approximation` and
:doc:`../theory/regression_estimators` the formulas, and :doc:`estimator_benchmark`
the measured errors.

Decision table
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 18 36

   * - Situation
     - ``estimator``
     - ``budget.fraction``
     - Why
   * - Up to about 10 players, or a reference to validate other runs
     - ``exact``
     - ``1.0`` (required)
     - Exact values; efficiency holds; :math:`2^n` generations
   * - You report the Banzhaf index
     - ``kernel_banzhaf``
     - ``per_player: 8`` or more
     - Regression on paired uniform samples; lowest Banzhaf error from
       :math:`4n` coalitions; ``mc`` is the sampler alternative
   * - Shapley on a longer prompt, scikit-learn available
     - ``proxy``
     - ``per_player: 8`` or more
     - Lowest Shapley error from :math:`8n` coalitions on the benchmark games;
       also computes Banzhaf; fits trees on every aggregation
   * - Shapley, values must add up to :math:`v(N) - v(\emptyset)`
     - ``leverage``
     - ``per_player: 4`` or more
     - Efficient at every budget by construction; second-lowest error from
       :math:`8n`; Shapley only
   * - Shapley, no extra dependency, payoffs cheap
     - ``mc``
     - ``per_player: 8`` or more
     - Every sample updates every player; no extra dependency
   * - Shapley, generous budget, payoff variance differs by coalition size
     - ``neyman_cc``
     - ``per_player: 16`` or more
     - Spends phase two where variance is high; better than ``complementary``
       from :math:`16n`
   * - Shapley and you want every coalition paired with its complement
     - ``complementary``
     - ``per_player: 16`` or more
     - Each pair :math:`(S, N \setminus S)` informs every player; pairs that
       cover every size cost about :math:`n \ln n` pairs
   * - A baseline any attribution method must beat
     - ``loo``
     - ``budget.evaluations: n + 2`` past ``max_players``
     - Leave-one-out, :math:`v(N) - v(N \setminus \{i\})`; :math:`n + 2`
       generations; not Shapley, so values need not add up
   * - Expensive backend (paid API, large model)
     - ``mc``
     - smallest that is stable
     - Re-run with two seeds; if the top players change, raise the budget

Runs whose budget plans more than :math:`2^{\text{max\_players}}` coalitions
(``budget.fraction`` times :math:`2^n`, or ``budget.evaluations``; every
full-table run on more than
``explanation.max_players`` players, default 24) are refused with
:exc:`~mllm_shap.errors.ConfigError` before any model call, whatever the
estimator; lower the fraction, use coarser players, or raise the limit
deliberately.

Whatever you choose, ``base_value`` and ``grand_value`` are evaluated exactly, and
with ``identity`` normalization the values sum to ``grand_value - base_value``
only on the full table (``exact``, or any estimator at ``fraction: 1.0``). Below
the full table the sum drifts and converges as the budget grows.

What a budget costs
-------------------

``budget.fraction`` scales the :math:`2^n` coalitions of the full table. The
snippet counts the coalitions each estimator requests at a 10 % budget:

.. code-block:: python

   from mllm_shap import (
       ComplementaryEstimator,
       ExactEstimator,
       MonteCarloEstimator,
       NeymanEstimator,
       PlayerSet,
   )

   print(f"{'players':>7} {'exact':>6} {'mc':>5} {'cc':>5} {'neyman':>6}")
   for n in (6, 8, 10, 12):
       players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(n)))
       counts = [
           sum(1 for _ in estimator.sample_masks(players, fraction, False, 0))
           for estimator, fraction in (
               (ExactEstimator(), 1.0),
               (MonteCarloEstimator(), 0.1),
               (ComplementaryEstimator(), 0.1),
               (NeymanEstimator(), 0.1),
           )
       ]
       print(f"{n:>7} {counts[0]:>6} {counts[1]:>5} {counts[2]:>5} {counts[3]:>6}")

.. guide-result:: choosing_budget_counts

Neyman's count is its phase one; the runner adds the phase-two coalitions up to
the same budget. Deduplication and the base generation reduce the number of
backend calls further; ``output.metrics.executed`` reports the real number.

Above about 25 players any fraction of :math:`2^n` you would type is more
coalitions than a backend can serve. ``budget.evaluations`` sets the count
instead: ``evaluations: 2000`` plans 2000 coalitions, the empty and grand ones
included, at any number of players, and ``budget.per_player: 4`` plans
:math:`\lceil 4n \rceil` of them. Each estimator still applies its floor (2 for
``mc``; :math:`2n + 2` for the paired estimators, which also round an odd count
down to whole pairs) and the ceiling :math:`2^n`; ``exact`` accepts only
``fraction: 1.0``. Set one of the three keys.

.. code-block:: yaml

   explanation:
     estimator: leverage
     budget:
       per_player: 4

Accuracy versus budget
----------------------

:doc:`estimator_benchmark` lists the error of each estimator against the exact
values of five ten-player games at :math:`2n` to :math:`32n` coalitions. The ranking
depends on the game; ``examples/estimator_comparison.ipynb`` (on :doc:`../examples`)
reruns it, and :ref:`theory-choosing-a-budget` says how to adapt it to your prompts.

Checklist
---------

1. Count players: ``len(TokenPartitioner().partition(snapshot).player_ids)``.
2. If :math:`2^n` generations are affordable, use ``exact``.
3. Otherwise start with ``proxy`` or ``leverage`` (``kernel_banzhaf`` for Banzhaf)
   at ``per_player: 8``, run two seeds and compare the ranking of the top players.
4. Raise the budget while the ranking still moves between seeds.
5. Record the estimator and budget with the results; the run manifest already
   does (:doc:`pipeline`).
