How a run works
===============

:class:`~mllm_shap.ExplainRunner` wires partitioning, masking, generation, scoring
and aggregation into one call. This page follows a run from the snapshot to the
attributions and explains the counters, the archive and the hooks you can attach.

Quick start
-----------

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
       Turn(messages=(Message(role=Role.USER, text="hello world"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "estimand": "shapley",
           "value_fn": "tfidf_cosine",
           "normalizer": "identity",
           "absence_policy": "pad",
           "archive": {"path": ""},
       },
   })

   output = ExplainRunner(config).explain_sync(snapshot)
   print("values:", tuple(round(value, 4) for value in output.result.values))
   print("metrics:", output.metrics)

.. guide-result:: pipeline_quick_start

The eight stages
----------------

1. **Partition.** The ``explanation.players`` plugin splits the snapshot into a
   :class:`~mllm_shap.PlayerSet`; with ``tokens`` every whitespace token of every
   message is one player.
2. **Resolve plugins.** The registry builds the estimator, estimand aggregator,
   value function, normalizer, absence policy and backend named in the config.
3. **Base generation.** The backend answers the unmasked prompt once. This text is
   the reference every coalition output is scored against, and it is the output
   of the grand coalition :math:`N`.
4. **Sample coalitions.** The estimator draws coalition masks for
   the budget (``fraction``, ``evaluations`` or ``per_player``) and ``seed``. The runner adds the empty and the grand
   coalition when the estimator did not draw them, so ``base_value`` and
   ``grand_value`` are always known.
5. **Render and key.** The absence policy turns each mask into a masked snapshot
   (``delete``, ``pad`` or ``neutral``). Each coalition gets a deduplication key
   and, for backends with a prefix cache, a prefix hash used to group jobs.
6. **Generate.** :class:`~mllm_shap.runtime.InferenceScheduler` runs the coalition
   prompts with at most ``max_inflight`` concurrent backend calls. The grand
   coalition is served from the base generation; repeated coalitions are served
   from the hot store or skipped by deduplication. Each output is appended to the
   archive.
7. **Score.** The value function compares every coalition output with the base
   generation and returns the payoff :math:`v(S)`. Corpus-fitted value functions
   (``tfidf_cosine``) are fitted once on all outputs of the run first. Two-phase
   estimators (``neyman_cc``) see the phase-one payoffs here, draw their
   phase-two coalitions, and stages 5 to 7 run again for those.
8. **Aggregate.** The estimator turns masks and payoffs into attributions with
   the estimand aggregator, the normalizer rescales them for display, and the run
   returns an :class:`~mllm_shap.ExplainRunOutput`.

Run output
----------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Meaning
   * - ``result``
     - :class:`~mllm_shap.ExplainResult`: ``estimand``, per-player ``values``,
       ``base_value`` (:math:`v(\emptyset)`) and ``grand_value`` (:math:`v(N)`)
   * - ``metrics``
     - :class:`~mllm_shap.runtime.scheduler.SchedulerMetrics` counters (below)
   * - ``run_id``
     - Snapshot id prefix plus a digest of the configuration and model id
   * - ``manifest``
     - :class:`~mllm_shap.RunManifest` written to the archive
   * - ``perf``
     - :class:`~mllm_shap.runtime.metrics.PerfSummary` wall-clock split between
       generation, scoring and aggregation

The scheduler counters answer "how many model calls did this cost":

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Counter
     - Meaning
   * - ``requested``
     - Coalition jobs submitted, one per sampled coalition
   * - ``executed``
     - Jobs that called the backend
   * - ``deduplicated``
     - Jobs skipped because an identical coalition key had already run
   * - ``cache_hits``
     - Jobs served from the in-memory hot store (the grand coalition is always
       one of them)
   * - ``kv_cache_hits``
     - Backend generations that reused a cached prompt prefix (transformers
       backend with ``kv_cache.enabled: true``; the cache is off by default)
   * - ``kv_cache_reused_tokens``
     - Prompt tokens those hits took from the cache instead of recomputing

Archives and reanalysis
-----------------------

With ``explanation.archive.path`` set (default ``./runs/{run_id}``) every run
writes a directory holding ``manifest.json``, the base generation and the
coalition outputs in ``archive.sqlite``. :meth:`~mllm_shap.ExplainRunner.reanalyze`
reads such a directory and re-scores the stored outputs without calling the
model, for example with a different normalizer:

.. code-block:: python

   from pathlib import Path

   archived = config.model_copy(
       update={
           "explanation": config.explanation.model_copy(
               update={"archive": config.explanation.archive.model_copy(
                   update={"path": "runs/{run_id}"},
               )},
           ),
       },
   )
   first = ExplainRunner(archived).explain_sync(snapshot)
   root = Path("runs") / first.run_id
   print("archive files:", sorted(path.name for path in root.iterdir()))

   rescored = archived.model_copy(
       update={
           "explanation": archived.explanation.model_copy(
               update={"normalizer": "min_max"},
           ),
       },
   )
   again = ExplainRunner(rescored).reanalyze(root)
   print("explain:  ", tuple(round(value, 4) for value in first.result.values))
   print("min_max:  ", tuple(round(value, 4) for value in again.result.values))
   print("backend calls during reanalyze:", again.metrics.executed)

.. guide-result:: pipeline_reanalyze

``reanalyze`` re-aggregates with the archived estimator and raises
:exc:`~mllm_shap.errors.ReanalyzeMismatchError` when the runner's estimator,
absence policy or players differ from the archived run. See :doc:`runtime` for
the archive format and the ``on_exists`` policy.

Progress and telemetry
----------------------

Pass ``progress`` to receive coalition counts (for a progress bar) and
``telemetry`` to time each stage. Both are optional and cost nothing when
omitted:

.. code-block:: python

   from mllm_shap.runtime.telemetry import InMemoryObservabilitySink

   class PrintProgress:
       def on_coalitions_planned(self, total: int) -> None:
           print(f"planned {total} coalitions")

       def on_coalition_finished(self, done: int, total: int) -> None:
           if done == total:
               print(f"finished {done}/{total}")

   sink = InMemoryObservabilitySink()
   runner = ExplainRunner(config, telemetry=sink, progress=PrintProgress())
   runner.explain_sync(snapshot)
   print("stages:", sorted({span.name for span in sink.spans}))

.. guide-result:: pipeline_progress

``explain`` and ``explain_sync``
--------------------------------

:meth:`~mllm_shap.ExplainRunner.explain` is a coroutine;
:meth:`~mllm_shap.ExplainRunner.explain_sync` runs it on a fresh event loop. Inside
an event loop that is already running (Jupyter notebooks, async web handlers)
``explain_sync`` raises :exc:`RuntimeError`; await the coroutine there instead:

.. code-block:: python

   output = await runner.explain(snapshot)

Reusing a backend
-----------------

By default every ``explain`` call builds the backend named in ``config.backend``
and closes it when the call ends. To keep one loaded model or HTTP client across
many prompts, pass ``backend=`` to the runner and close it yourself:

.. code-block:: python

   from mllm_shap.backends.mock import MockBackend

   runner = ExplainRunner(config, backend=MockBackend(model_id="stub"))
   prompts = ("hello world", "goodbye world")
   for text in prompts:
       prompt = ConversationSnapshot.from_turns((
           Turn(messages=(Message(role=Role.USER, text=text),)),
       ))
       values = runner.explain_sync(prompt).result.values
       print(text, "->", tuple(round(value, 4) for value in values))

.. guide-result:: pipeline_reuse_backend

In async code call ``await runner.aclose()`` when done; it closes the injected
backend if it holds resources.

The mock backend is deterministic and needs no GPU or optional extras; for model
servers see :doc:`backends` and for every configuration key see :doc:`config`.

Notebooks
---------

``examples/explain_pipeline.ipynb`` walks through one run from the snapshot to
the JSON result, with every estimator and estimand and the monitoring hooks;
``examples/archive_reanalyze.ipynb`` covers archives and ``reanalyze`` (both
embedded on :doc:`../examples`).
