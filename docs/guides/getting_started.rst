Getting started
===============

This page gets you from zero to a first token attribution in a few minutes, and
shows the two numbers every Shapley explanation is checked against.

Installation
------------

Install from PyPI (Python 3.12 or newer):

.. code-block:: bash

   pip install nlp-shap

From a local clone (development uses `uv <https://docs.astral.sh/uv/>`_):

.. code-block:: bash

   git clone https://github.com/Pawlo77/nlp-shap
   cd nlp-shap
   make install

Your first aggregation
----------------------

The estimand aggregators take boolean **coalition masks** and aligned payoffs
:math:`v(S)`. On a three-player majority game, Shapley and Banzhaf disagree:

.. code-block:: python

   from itertools import product

   from nlp_shap import BanzhafAggregator, ShapleyAggregator

   num_players = 3
   masks = [tuple(bits) for bits in product([False, True], repeat=num_players)]

   def majority_payoff(mask: tuple[bool, ...]) -> float:
       return 1.0 if sum(mask) >= 2 else 0.0

   payoffs = [majority_payoff(mask) for mask in masks]

   shapley = ShapleyAggregator().aggregate(masks, payoffs)
   banzhaf = BanzhafAggregator().aggregate(masks, payoffs)

   print("Shapley:", shapley)
   print("Banzhaf:", banzhaf)

.. guide-result:: getting_started_majority

See :doc:`estimands` for why the two differ.

Explain a short prompt
----------------------

For end-to-end token attributions, build a
:class:`~nlp_shap.ConversationSnapshot`, validate an
:class:`~nlp_shap.ExplainConfig`, and call :class:`~nlp_shap.ExplainRunner`.
The ``mock`` backend needs no model, GPU or optional extra:

.. code-block:: python

   import math

   from nlp_shap import (
       ConversationSnapshot,
       ExplainConfig,
       ExplainRunner,
       Message,
       Role,
       Turn,
   )

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="refund my order"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "value_fn": "tfidf_cosine",
           "absence_policy": "pad",
           "archive": {"path": ""},  # keep this demo off disk
       },
   })
   output = ExplainRunner(config).explain_sync(snapshot)
   result = output.result

   print("values:", tuple(round(value, 4) for value in result.values))
   print("base_value v(empty):", round(result.base_value, 4))
   print("grand_value v(N):", round(result.grand_value, 4))
   # Efficiency: Shapley values sum to v(N) - v(empty).
   print(
       "efficient:",
       math.isclose(sum(result.values), result.grand_value - result.base_value),
   )

.. guide-result:: getting_started_explain

Each token of ``"refund my order"`` is a player. ``base_value`` is the payoff of
the fully masked prompt and ``grand_value`` the payoff of the full prompt; Shapley
values distribute the difference. With the default ``identity`` normalizer the
check above holds for every exact run and is the first thing to verify when an
attribution looks wrong.

Without ``archive.path: ""`` the run is archived under ``./runs/{run_id}``, and
re-running the same configuration on the same snapshot raises
:exc:`~nlp_shap.errors.ArchiveExistsError` unless
``explanation.archive.on_exists`` is ``overwrite``. See :doc:`config`.

Which estimator?
----------------

``exact`` scores all :math:`2^n` coalitions and is only practical for short
prompts. For longer inputs pick ``mc``, ``complementary`` or ``neyman_cc`` and a
``budget.fraction`` below 1; :doc:`choosing_an_estimator` has the decision table
and :doc:`approximation` the code.

Label results for archives
--------------------------

Every run carries a :class:`~nlp_shap.RunManifest` that records the estimand,
estimator, value function and players. It round-trips through JSON:

.. code-block:: python

   from nlp_shap import parse_manifest

   restored = parse_manifest(output.manifest.to_dict())
   print("schema_version:", restored.schema_version)
   print("estimand:", restored.estimand.value)
   print("estimator:", restored.estimator)
   print("players:", restored.player_ids)

.. guide-result:: getting_started_manifest

Logging
-------

Importing ``nlp_shap`` leaves your logging configuration alone. Call
``nlp_shap.configure_logging(level="INFO")`` to see the pipeline's stage events
(run start, base generation, scheduled coalitions and run duration) on stderr,
``nlp_shap.configure_logging()`` to apply the ``[tool.logging]`` section of the
nearest ``pyproject.toml`` (or pass a path), or set ``NLP_SHAP_LOGGING_CONFIG``.
:doc:`runtime` lists the events.

Next steps
----------

- :doc:`pipeline` — what happens inside a run, metrics and archives
- :doc:`cli` — the same run from the command line, with JSON input and output
- :doc:`config` — every configuration key
- :doc:`choosing_an_estimator` — exact, Monte Carlo, complementary or Neyman
- :doc:`backends` — run against LM Studio, Hugging Face or an HTTP API
- :doc:`../examples` — runnable notebooks
