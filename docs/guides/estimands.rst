Using estimand aggregators
==========================

The :mod:`mllm_shap.estimation.estimands` package provides **estimand aggregators**
that turn coalition masks and payoffs :math:`v(S)` into per-player attributions.
Each aggregator reports which estimand it implements, and every run records that
label, so Banzhaf output is never reported as Shapley. The majority-game example
in :doc:`getting_started` shows the two disagreeing on the same coalitions.

Coalition weights
-----------------

Inspect the weight each estimand gives a marginal contribution
:math:`v(S \cup \{i\}) - v(S)` with :math:`|S| = k`:

.. code-block:: python

   from mllm_shap import BanzhafAggregator, ShapleyAggregator

   shapley = ShapleyAggregator()
   banzhaf = BanzhafAggregator()

   n = 4
   for k in range(n):
       print(
           f"k={k}",
           round(shapley.coalition_weight(k, n), 6),
           round(banzhaf.coalition_weight(k, n), 6),
       )

.. guide-result:: estimands_coalition_weights

Shapley weights sum to one over all coalitions of each player, which is what makes
the values efficient; Banzhaf weights are uniform.

Choosing the estimand of a run
------------------------------

``explanation.estimand`` selects the aggregator; the same coalitions and payoffs
are used either way. The result and the manifest carry the label:

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
       Turn(messages=(Message(role=Role.USER, text="cancel my old plan"),)),
   ))
   for estimand in ("shapley", "banzhaf"):
       config = ExplainConfig.model_validate({
           "backend": {"kind": "mock", "model_id": "stub"},
           "explanation": {
               "estimator": "exact",
               "estimand": estimand,
               "absence_policy": "pad",
               "archive": {"path": ""},
           },
       })
       output = ExplainRunner(config).explain_sync(snapshot)
       print(
           output.result.estimand.value,
           output.manifest.estimand.value,
           tuple(round(value, 4) for value in output.result.values),
       )

.. guide-result:: estimands_run_label

Wire values for JSON manifests use :data:`~mllm_shap.domain.estimands.EstimandWire`
(``"shapley"`` | ``"banzhaf"``).

Partial samples
---------------

With a budget below the full table the aggregators see only part of the coalition
table. Both are then estimated from the sampled coalitions stratified by size:
the mean payoff with a player minus the mean without it, per size. Shapley
weights the sizes equally, Banzhaf by their binomial probabilities, each
renormalized over the sizes that have both means. Both converge as coverage
grows, but efficiency holds exactly only on the full table (and for ``leverage``). :ref:`theory-partial-samples` gives the formulas and
:doc:`choosing_an_estimator` the practical consequences.

Notebook
--------

``examples/estimands_toy_game.ipynb`` (on :doc:`../examples`) walks through the
majority and additive games step by step.

Further reading
---------------

- Theory: :doc:`../theory/shapley_values`, :doc:`../theory/estimands`
- Cooperative games: :doc:`../theory/cooperative_games`
- API: :doc:`../api/estimation`
