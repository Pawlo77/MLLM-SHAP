Extending mllm-shap
===================

Every stage of a run is a plugin looked up by name in a
:class:`~mllm_shap.PluginRegistry`: the partitioner, absence policy, estimator,
estimand, value function, normalizer, backend and renderer. Third-party packages
add plugins through entry points; scripts and tests register them directly. This
page writes one plugin of each common kind and runs it. The snippets share one
namespace.

Optional extras
---------------

Install only what your deployment needs:

.. code-block:: bash

   pip install mllm-shap                    # core only (no torch)
   pip install "mllm-shap[transformers]"    # Hugging Face text backend
   pip install "mllm-shap[lmstudio]"        # LM Studio SDK backend
   pip install "mllm-shap[api]"             # OpenAI-compatible HTTP backend
   pip install "mllm-shap[viz]"             # matplotlib token attribution charts
   pip install "mllm-shap[proxy]"           # scikit-learn: the ProxySHAP estimator
   pip install "mllm-shap[audio]"           # audio decoding and alignment
   pip install "mllm-shap[semantic]"        # encoders of embedding_cosine, bertscore,
                                            # nli_logodds and semascore
   pip install "mllm-shap[audio-semantic]"  # speech_bertscore and transcribed

How plugins are built
---------------------

A plugin is registered as a **factory**: any callable that takes a
:class:`~mllm_shap.plugins.context.PluginContext` and returns the plugin. A class
whose ``__init__`` takes the context is the usual factory. The context carries

- ``config``: the validated :class:`~mllm_shap.ExplainConfig` of the run;
- ``options``: ``explanation.plugin_options["<group>/<name>"]`` or
  ``explanation.plugin_options["<name>"]``, empty when absent;
  :meth:`~mllm_shap.plugins.context.PluginContext.option` reads one with a typed
  default. A plugin class that sets ``options_model`` to a
  :class:`~mllm_shap.protocols.plugin.PluginOptions` subclass gets its options
  validated first (unknown keys and wrong types raise
  :exc:`~mllm_shap.errors.ConfigError`); without it the options pass unchecked;
- ``registry``: the registry that built the plugin.

A custom value function with options
------------------------------------

This value function scores a coalition output by how much of the base
generation's vocabulary it keeps, optionally ignoring short words. The threshold
comes from ``plugin_options``:

.. code-block:: python

   from mllm_shap import (
       ConversationSnapshot,
       ExplainConfig,
       ExplainRunner,
       Message,
       PluginGroup,
       PluginRegistry,
       Role,
       Turn,
   )
   from mllm_shap.plugins import PluginContext, register_builtin_plugins
   from mllm_shap.protocols.backend import GenerationResult
   from mllm_shap.protocols.plugin import PluginOptions


   class VocabularyOverlapOptions(PluginOptions):
       min_length: int = 1


   class VocabularyOverlapValue:
       options_model = VocabularyOverlapOptions

       def __init__(self, context: PluginContext) -> None:
           self._min_length = context.option("min_length", 1)

       @property
       def name(self) -> str:
           return "vocab_overlap"

       def _words(self, result: GenerationResult) -> set[str]:
           return {w for w in result.text.split() if len(w) >= self._min_length}

       def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
           reference = self._words(base)
           if not reference:
               return 0.0
           return len(reference & self._words(candidate)) / len(reference)


   registry = PluginRegistry()
   register_builtin_plugins(registry)
   registry.register(PluginGroup.VALUE_FNS, "vocab_overlap", VocabularyOverlapValue)

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="where is my parcel"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "value_fn": "vocab_overlap",
           "absence_policy": "pad",
           "archive": {"path": ""},
           "plugin_options": {"vocab_overlap": {"min_length": 4}},
       },
   })
   output = ExplainRunner(config, registry=registry).explain_sync(snapshot)
   print(tuple(round(value, 4) for value in output.result.values))

.. guide-result:: extending_value_fn

A value function satisfies :class:`~mllm_shap.protocols.value.ValueFunction`; add
a ``fit(corpus)`` method
(:class:`~mllm_shap.protocols.value.FittableValueFunction`) when it needs
statistics over all outputs of the run before scoring, as ``tfidf_cosine`` does.

A custom backend
----------------

A backend needs a ``model_id`` property and an async ``generate(snapshot,
generation)`` that returns an object with a ``text`` attribute.
:func:`~mllm_shap.value.generation_records.generation_record_from_text` builds the
record the built-in value functions expect. This one answers with the prompt's
words in reverse order:

.. code-block:: python

   from mllm_shap.pipeline.config import GenerationConfig
   from mllm_shap.value.generation_records import generation_record_from_text


   class ReverseBackend:
       def __init__(self, context: PluginContext) -> None:
           self._model_id = context.config.backend.model_id

       @property
       def model_id(self) -> str:
           return self._model_id

       async def generate(
           self, snapshot: ConversationSnapshot, generation: GenerationConfig
       ) -> GenerationResult:
           words = " ".join(m.text for t in snapshot.turns for m in t.messages)
           reply = " ".join(reversed(words.split()))[: generation.max_new_tokens]
           return generation_record_from_text(reply)


   registry.register(PluginGroup.BACKENDS, "reverse", ReverseBackend)
   reverse_config = config.model_copy(
       update={"backend": config.backend.model_copy(update={"kind": "reverse"})}
   )
   output = ExplainRunner(reverse_config, registry=registry).explain_sync(snapshot)
   print(tuple(round(value, 4) for value in output.result.values))

.. guide-result:: extending_backend

Backends that hold clients or models may also implement ``async aclose()``
(:class:`~mllm_shap.protocols.backend.SupportsAclose`); the runner calls it after
each run. Settings only your backend understands belong under
``backend.options``.

A custom estimator
------------------

An estimator chooses which coalitions to evaluate and turns their payoffs into
attributions with the estimand aggregator of the run. This leave-one-out
estimator evaluates the full prompt and every prompt with one player removed.
Fed to the Shapley aggregator, the stratified estimate it gets from those pairs
is exactly the occlusion score :math:`v(N) - v(N \setminus \{i\})`:

.. code-block:: python

   from collections.abc import Iterator, Sequence

   from mllm_shap import CoalitionMask, PlayerSet
   from mllm_shap.protocols.estimand import EstimandAggregator


   class LeaveOneOutEstimator:
       def __init__(self, context: PluginContext | None = None) -> None:
           _ = context

       @property
       def name(self) -> str:
           return "leave_one_out"

       def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
           _ = snapshot

       def sample_masks(
           self,
           player_set: PlayerSet,
           budget_fraction: float,
           include_minimal_masks: bool,
           seed: int,
       ) -> Iterator[CoalitionMask]:
           n = player_set.num_players
           yield CoalitionMask.from_sequence((True,) * n)
           for i in range(n):
               yield CoalitionMask.from_sequence(tuple(j != i for j in range(n)))

       def estimate_attributions(
           self,
           masks: Sequence[CoalitionMask],
           payoffs: Sequence[float],
           aggregator: EstimandAggregator,
       ) -> list[float]:
           return aggregator.aggregate([mask.present for mask in masks], payoffs)


   registry.register(PluginGroup.ESTIMATORS, "leave_one_out", LeaveOneOutEstimator)
   loo_config = reverse_config.model_copy(
       update={
           "explanation": reverse_config.explanation.model_copy(
               update={"estimator": "leave_one_out"},
           ),
       },
   )
   output = ExplainRunner(loo_config, registry=registry).explain_sync(snapshot)
   print("coalitions:", output.metrics.requested)
   print(tuple(round(value, 4) for value in output.result.values))

.. guide-result:: extending_estimator

The runner adds the empty coalition, so ``base_value`` is still reported; the
leave-one-out values are not efficient. Estimators that need payoffs before
drawing more coalitions implement the two-phase protocol
:class:`~mllm_shap.protocols.estimator.TwoPhaseEstimator`, as ``neyman_cc`` does.

A custom partitioner
--------------------

A partitioner decides what a player is. This one makes every message one player,
the shape of a retrieval-augmented prompt where each retrieved passage is a
message:

.. code-block:: python

   from mllm_shap import PlayerSpan


   class MessagePartitioner:
       def __init__(self, context: PluginContext | None = None) -> None:
           _ = context

       @property
       def name(self) -> str:
           return "messages"

       def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
           messages = snapshot.messages()
           return PlayerSet(
               player_ids=tuple(
                   f"{snapshot.snapshot_id}:{index}" for index in range(len(messages))
               ),
               spans=tuple(
                   PlayerSpan(index, 0, len(message.text), message.text)
                   for index, message in enumerate(messages)
               ),
           )


   passages = ConversationSnapshot.from_turns((
       Turn(messages=(
           Message(role=Role.SYSTEM, text="Refunds take 14 days."),
           Message(role=Role.SYSTEM, text="Shipping is free over 50 EUR."),
           Message(role=Role.USER, text="How long do refunds take?"),
       )),
   ))
   print(MessagePartitioner().partition(passages).num_players)

.. guide-result:: extending_partitioner

Register it under :attr:`PluginGroup.PARTITIONS
<mllm_shap.plugins.groups.PluginGroup.PARTITIONS>`. Each
:class:`~mllm_shap.domain.players.PlayerSpan` names the message (turns flattened)
and the character range the player covers; the built-in absence policies delete,
pad or blank exactly those ranges and keep all other text, so any partitioner
works with ``delete``, ``pad`` and ``neutral``.

Testing a plugin
----------------

Test a plugin the way the pipeline uses it: register it, resolve it by name with
a context, and assert on behaviour.

.. code-block:: python

   context = PluginContext(config=config, registry=registry)
   value_fn = registry.resolve(PluginGroup.VALUE_FNS, "vocab_overlap", context)
   base = generation_record_from_text("refund issued today")
   print(value_fn.score(base, generation_record_from_text("refund issued")))

.. guide-result:: extending_test

``resolve`` passes ``explanation.plugin_options["value_fns/<name>"]`` (or the bare
``<name>`` key) as the context options, validated against ``options_model``, so the
plugin above sees ``min_length: 4``.

Publishing plugins
------------------

Register plugins under ``project.entry-points`` in your package's
``pyproject.toml``; :class:`~mllm_shap.ExplainRunner` loads entry points of every
group when it builds its default registry:

.. code-block:: toml

   [project.entry-points."mllm_shap.value_fns"]
   vocab_overlap = "my_package.values:VocabularyOverlapValue"

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Group
     - Purpose
   * - ``mllm_shap.estimators``
     - Coalition sampling strategies (``exact``, ``mc``, …)
   * - ``mllm_shap.estimands``
     - Payoff aggregation (``shapley``, ``banzhaf``)
   * - ``mllm_shap.value_fns``
     - Coalition scoring utilities (``tfidf_cosine``, ``logprob``, …)
   * - ``mllm_shap.normalizers``
     - Post-aggregation scaling (``identity``, ``min_max``, …)
   * - ``mllm_shap.partitions``
     - Player definitions (``tokens``, …)
   * - ``mllm_shap.absence_policies``
     - Masked snapshot rendering (``delete``, ``pad``, …)
   * - ``mllm_shap.backends``
     - Model inference connectors (``mock``, ``transformers``, …)
   * - ``mllm_shap.renderers``
     - Attribution visualization (``token_text``, ``token_bar``)

Names already registered are skipped, so a third-party plugin cannot silently
replace a built-in one; call ``registry.register(..., replace=True)`` explicitly
for that.

Further reading
---------------

- :doc:`../api/plugins` — registry API and built-in groups
- :doc:`../api/protocols` — the protocols each plugin kind satisfies
- :doc:`config` — ``plugin_options`` and every other key
- ``examples/plugins_extending.ipynb`` (embedded on :doc:`../examples`) — the
  registry, ``plugin_options`` validation and a custom backend and value function
