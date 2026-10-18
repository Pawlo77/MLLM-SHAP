Using approximate estimators
============================

Approximate estimators sample coalitions under a budget and aggregate payoffs into
attributions. Monte Carlo delegates to estimand plugins; complementary and Neyman
estimators use complementary-contribution (CC) aggregation. Every
estimator takes the same ``estimate_attributions(masks, payoffs, aggregator)``
call; CC estimators accept the Shapley aggregator only. See
:doc:`../theory/approximation` for what each estimator computes from a partial
sample.

A toy game with a known answer
------------------------------

The snippets on this page share one five-player game: each player adds its weight,
and players ``p0`` and ``p1`` earn a bonus of 2 when both are present. The exact
Shapley values are the weights, plus half the bonus for ``p0`` and ``p1``:

.. code-block:: python

   from mllm_shap import (
       CoalitionMask,
       ComplementaryEstimator,
       ExactEstimator,
       MonteCarloEstimator,
       NeymanEstimator,
       PlayerSet,
       ShapleyAggregator,
   )

   player_set = PlayerSet(player_ids=("p0", "p1", "p2", "p3", "p4"))
   weights = (1.0, 0.5, 2.0, -1.0, 0.25)

   def payoff(mask: CoalitionMask) -> float:
       additive = sum(w for w, present in zip(weights, mask.present) if present)
       bonus = 2.0 if mask.present[0] and mask.present[1] else 0.0
       return additive + bonus

   exact = ExactEstimator()
   all_masks = list(exact.sample_masks(player_set, 1.0, False, seed=0))
   reference = exact.estimate_attributions(
       all_masks, [payoff(mask) for mask in all_masks], ShapleyAggregator()
   )
   print("coalitions:", len(all_masks))
   print("exact:", tuple(round(value, 4) for value in reference))

.. guide-result:: approximation_reference

Monte Carlo
-----------

.. code-block:: python

   estimator = MonteCarloEstimator()
   masks = list(estimator.sample_masks(
       player_set,
       budget_fraction=0.5,
       include_minimal_masks=True,
       seed=42,
   ))
   values = estimator.estimate_attributions(
       masks,
       [payoff(mask) for mask in masks],
       ShapleyAggregator(),
   )
   print("samples:", len(masks))
   print("mc:", tuple(round(value, 4) for value in values))

.. guide-result:: approximation_monte_carlo

The same masks aggregated with :class:`~mllm_shap.BanzhafAggregator` give the
Banzhaf estimate instead; Monte Carlo and ``proxy`` support both estimands,
``kernel_banzhaf`` Banzhaf only.

Complementary pairs
-------------------

.. code-block:: python

   estimator = ComplementaryEstimator()
   masks = list(estimator.sample_masks(
       player_set,
       budget_fraction=0.5,
       include_minimal_masks=True,
       seed=7,
   ))
   # The (N, empty) pair comes first, then the singleton pairs ({i}, N\{i}),
   # then pairs S, N\S. Order does not matter for aggregation: masks are
   # matched to their complements by value.
   values = estimator.estimate_attributions(
       masks,
       [payoff(mask) for mask in masks],
       ShapleyAggregator(),
   )
   print("samples:", len(masks))
   print("complementary:", tuple(round(value, 4) for value in values))

.. guide-result:: approximation_complementary

Neyman two-phase sampling
-------------------------

Neyman allocation needs phase-one payoffs before drawing phase-two masks:

.. code-block:: python

   estimator = NeymanEstimator(initial_num_samples=2, use_standard_method=False)
   phase_one = list(estimator.sample_masks(
       player_set,
       budget_fraction=0.8,
       include_minimal_masks=False,
       seed=11,
   ))
   payoffs_one = [payoff(mask) for mask in phase_one]
   estimator.begin_allocation(phase_one, payoffs_one)

   phase_two = list(estimator.sample_allocation_masks())
   payoffs_two = [payoff(mask) for mask in phase_two]

   values = estimator.estimate_attributions(
       phase_one + phase_two,
       payoffs_one + payoffs_two,
       ShapleyAggregator(),
   )
   print("phase one:", len(phase_one), "phase two:", len(phase_two))
   print("neyman:", tuple(round(value, 4) for value in values))

.. guide-result:: approximation_neyman

:class:`~mllm_shap.ExplainRunner` runs both phases for you when
``explanation.estimator`` is ``neyman_cc``; the phase-one size comes from the
``neyman`` config section.

Configuration
-------------

Set the estimator and budget in :class:`~mllm_shap.ExplainConfig`:

.. code-block:: yaml

   explanation:
     estimator: neyman_cc
     budget:
       fraction: 0.8
     include_minimal_masks: false
     neyman:
       initial_num_samples: 2
       use_standard_method: false

Phase one is sized by ``neyman.initial_num_samples`` or, when that is unset,
``neyman.initial_fraction``; with neither set a default based on the number of
players is used. On small budgets phase one can use the whole allowance; the
estimator then aggregates phase one alone; up to :math:`2 + 2n` masks it evaluates
the same coalitions as complementary sampling, and above that its error is often
close to it (``examples/estimator_comparison.ipynb`` shows a game where it is
larger). :doc:`config` lists every key.

Plugin resolution
-----------------

Estimators are registered under the ``mllm_shap.estimators`` entry-point group;
the names are the values ``explanation.estimator`` accepts:

.. code-block:: python

   from mllm_shap import PluginGroup, PluginRegistry
   from mllm_shap.plugins import register_builtin_plugins

   registry = PluginRegistry()
   register_builtin_plugins(registry)
   print(registry.names(PluginGroup.ESTIMATORS))

.. guide-result:: approximation_plugins

Regression estimators and long prompts
--------------------------------------

``leverage`` (Shapley), ``kernel_banzhaf`` (Banzhaf) and ``proxy`` (both; extra
``mllm-shap[proxy]``) fit a regression to the masks and payoffs, so a budget of a
few multiples of :math:`n` suffices (:doc:`../theory/regression_estimators`). Give
them ``budget.per_player``. For prompts of many paragraphs, ``players: sentences``
with ``explain_hierarchical`` attributes paragraphs first and only the sentences
of the top ones afterwards:

.. code-block:: python

   from mllm_shap import (
       ConversationSnapshot, ExplainConfig, ExplainRunner, Message, Role, Turn,
   )

   text = "\n\n".join(
       " ".join(f"Point {p}.{k} is made here." for k in range(8)) for p in range(10)
   )
   snapshot = ConversationSnapshot.from_turns(
       (Turn(messages=(Message(role=Role.USER, text=text),)),)
   )
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "mock"},
       "generation": {"max_new_tokens": 8},
       "explanation": {
           "estimator": "leverage",
           "players": "sentences",
           "budget": {"per_player": 4},
           "hierarchical": {"keep_fraction": 0.2},
       },
   })
   output = ExplainRunner(config).explain_hierarchical_sync(snapshot)
   print("groups kept:", output.kept_groups)
   print("coarse calls:", output.coarse.metrics.requested)
   print("fine calls:", output.fine.metrics.requested, "(flat game: 320)")

.. guide-result:: approximation_hierarchical

Choosing an estimator
---------------------

.. list-table::
   :header-rows: 1

   * - Estimator
     - Best when
     - Estimand support
     - Accuracy at low budget
     - Partial-sample caveat
   * - Monte Carlo (``mc``)
     - You need Banzhaf, or Shapley at low and medium budgets
     - Shapley, Banzhaf
     - Next to the regression estimators in :doc:`estimator_benchmark`
     - Not efficient below the full table; strata without both means are skipped
   * - Complementary (``complementary``)
     - Every coalition should be scored together with its complement
     - Shapley only
     - Converges more slowly than Monte Carlo from a budget of :math:`8n`
     - Not efficient below the full table
   * - Neyman (``neyman_cc``)
     - Generous budgets where payoff variance differs by coalition size
     - Shapley only (two-phase)
     - Close to complementary while phase one uses the whole budget
     - Phase-two allocation needs enough phase-one samples per size
   * - ``leverage``, ``kernel_banzhaf``, ``proxy``
     - Budgets of :math:`8n` and more
     - Shapley; Banzhaf; both
     - Lowest error in :doc:`estimator_benchmark`
     - ``leverage`` is efficient at every budget; ``proxy`` needs scikit-learn

:doc:`choosing_an_estimator` turns this into a decision table, and
``examples/estimator_comparison.ipynb`` (on :doc:`../examples`) measures the error
of each estimator against the exact values as the budget grows.
