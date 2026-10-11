Choosing an estimator
=====================

Every coalition an estimator asks for is one model generation (minus the ones
served from the cache), so the estimator and ``budget.fraction`` decide what a run
costs. This page turns the trade-offs into a decision table. :doc:`approximation`
has the code for each estimator and :doc:`../theory/approximation` the formulas.

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
     - ``mc``
     - as budget allows
     - The only built-in sampler that feeds the Banzhaf aggregator
   * - Shapley on a longer prompt
     - ``mc``
     - 0.2 to 0.5
     - Lowest error from fraction 0.2 up on the benchmark game of
       :ref:`theory-choosing-a-budget`
   * - Shapley, generous budget, payoff variance differs by coalition size
     - ``neyman_cc``
     - 0.5 or more
     - Spends phase two where variance is high; below that phase one uses the
       whole budget and its error is close to ``complementary`` on the
       benchmark game
   * - Shapley and you want every coalition paired with its complement
     - ``complementary``
     - 0.3 or more
     - Each pair :math:`(S, N \setminus S)` informs every player; the singleton
       pairs come first, so every size-1 stratum is covered from a budget of
       :math:`2 + 2n` masks; converges more slowly than ``mc`` on the benchmark
       game
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
included, at any number of players. Each estimator still applies its floor (2
for ``mc``; :math:`2n + 2` for ``complementary`` and ``neyman_cc``, which also
round an odd count down to whole pairs) and the ceiling :math:`2^n`; ``exact``
accepts only ``fraction: 1.0``.

.. code-block:: yaml

   explanation:
     estimator: mc
     budget:
       evaluations: 2000

Accuracy versus budget
----------------------

:ref:`theory-choosing-a-budget` plots the error of each estimator against the
exact Shapley values of an eight-player game as the budget grows. The ranking
depends on the game; ``examples/estimator_comparison.ipynb`` (on
:doc:`../examples`) repeats the comparison on more games. Use them to pick a
budget for your number of players before running on an expensive backend.

Checklist
---------

1. Count players: ``len(TokenPartitioner().partition(snapshot).player_ids)``.
2. If :math:`2^n` generations are affordable, use ``exact``.
3. Otherwise start with ``mc`` at ``fraction: 0.2``, run two seeds and compare
   the ranking of the top players.
4. Raise the budget while the ranking still moves between seeds; at
   ``fraction: 0.5`` and above ``neyman_cc`` is a good alternative.
5. Record the estimator and budget with the results; the run manifest already
   does (:doc:`pipeline`).
