Using the exact estimator
=========================

:class:`~mllm_shap.estimation.exact.ExactEstimator` lazily enumerates every
coalition mask, including the empty and grand coalitions, and delegates
attribution to a registered estimand plugin.

Quick start
-----------

.. code-block:: python

   from itertools import product

   from mllm_shap import (
       BanzhafAggregator,
       CoalitionMask,
       ExactEstimator,
       PlayerSet,
       ShapleyAggregator,
   )

   player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
   estimator = ExactEstimator()

   masks = list(estimator.sample_masks(
       player_set,
       budget_fraction=1.0,
       include_minimal_masks=False,
       seed=0,
   ))
   print("coalitions:", len(masks))  # 8 for three players

   all_masks = [
       tuple(bits) for bits in product([False, True], repeat=player_set.num_players)
   ]

   def majority_payoff(mask: tuple[bool, ...]) -> float:
       return 1.0 if sum(mask) >= 2 else 0.0

   payoffs = [majority_payoff(mask) for mask in all_masks]
   coalition_masks = tuple(CoalitionMask.from_sequence(mask) for mask in all_masks)

   shapley = estimator.estimate_attributions(
       coalition_masks,
       payoffs,
       ShapleyAggregator(),
   )
   banzhaf = estimator.estimate_attributions(
       coalition_masks,
       payoffs,
       BanzhafAggregator(),
   )

   print("Shapley:", shapley)
   print("Banzhaf:", banzhaf)

.. guide-result:: exact_quick_start

Bitmask iteration
-----------------

For large player sets, iterate packed coalition integers and decode only when
needed:

.. code-block:: python

   from mllm_shap import ExactEstimator

   for index, mask_int in enumerate(ExactEstimator.iter_mask_ints(3)):
       present = ExactEstimator.mask_int_to_present(mask_int, 3)
       print(present)
       if index >= 2:
           break

.. guide-result:: exact_bitmask_iteration

Budget requirement
------------------

Exact enumeration requires a full budget fraction:

.. code-block:: python

   from mllm_shap import ExactEstimator, PlayerSet

   player_set = PlayerSet(player_ids=("p0",))
   try:
       ExactEstimator().sample_masks(
           player_set,
           budget_fraction=0.5,
           include_minimal_masks=False,
           seed=0,
       )
   except ValueError as exc:
       print(exc)

.. guide-result:: exact_budget_requirement

Plugin resolution
-----------------

The pipeline builds the estimator through the plugin registry. Every plugin
factory receives a :class:`~mllm_shap.plugins.context.PluginContext` holding the
run configuration:

.. code-block:: python

   from mllm_shap import ExplainConfig, PluginContext, PluginGroup, PluginRegistry
   from mllm_shap.plugins import register_builtin_plugins

   registry = PluginRegistry()
   register_builtin_plugins(registry)
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {"estimator": "exact"},
   })
   context = PluginContext(config=config, registry=registry)

   estimator = registry.resolve(
       PluginGroup.ESTIMATORS, config.explanation.estimator, context
   )
   print(type(estimator).__name__, estimator.name)

.. guide-result:: exact_plugin

Inside the pipeline
-------------------

With ``estimator: exact`` the runner requests all :math:`2^n` coalitions. The
grand coalition :math:`N` is the unmasked prompt, whose output the base
generation already produced, so it is served from the hot store instead of a
second backend call:

.. code-block:: python

   from mllm_shap import (
       ConversationSnapshot,
       ExplainConfig,
       ExplainRunner,
       Message,
       Role,
       Turn,
   )

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="is it in stock"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "absence_policy": "pad",
           "archive": {"path": ""},
       },
   })
   output = ExplainRunner(config).explain_sync(snapshot)
   print("players:", len(output.result.values))
   print("requested:", output.metrics.requested)
   print("served from the base generation:", output.metrics.cache_hits)
   print(
       "sum of values:",
       round(sum(output.result.values), 6),
       "= v(N) - v(empty):",
       round(output.result.grand_value - output.result.base_value, 6),
   )

.. guide-result:: exact_pipeline

Notebook
--------

See the runnable walkthrough in :doc:`../examples` —
``examples/exact_estimation.ipynb``.

Further reading
---------------

- :doc:`estimands` — Shapley and Banzhaf aggregators
- :doc:`config` — ``explanation.estimator`` key
- Theory: :doc:`../theory/exact`
- API: :doc:`../api/estimation`
