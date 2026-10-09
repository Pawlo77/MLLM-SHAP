Runtime core
============

The runtime layer is the execution substrate for coalition evaluations:

1. **Run archive** — SQLite index plus the generated text of every coalition.
2. **Dedup registry** — SHA-256 coalition keys that collapse repeated work.
3. **Hot store** — in-memory LRU of recent generations.
4. **Inference scheduler** — async execution with bounded ``max_inflight``.

Run archive
-----------

:class:`~nlp_shap.runtime.archive.RunArchive` persists
:class:`~nlp_shap.runtime.archive.CoalitionRecord` rows with the packed mask,
timing and cache-hit flag in ``archive.sqlite``; the generated text sits next to
it, so :meth:`~nlp_shap.runtime.archive.RunArchive.history_lazy` streams rows
without loading every generation into memory. The directory also holds
``manifest.json`` (:class:`~nlp_shap.RunManifest`) and the base generation.
Archives store generations, not payoffs: ``reanalyze`` recomputes payoffs and
attributions from them. :meth:`~nlp_shap.runtime.archive.RunArchive.open`
creates an archive and refuses an existing one unless ``on_exists`` is
``"overwrite"``; :meth:`~nlp_shap.runtime.archive.RunArchive.load` opens one for
reading and checks its schema version.

Coalition deduplication
-----------------------

:func:`~nlp_shap.runtime.dedup.build_coalition_key` hashes snapshot id, player
order, packed mask bytes, absence policy, model id, and generation settings.
:class:`~nlp_shap.runtime.dedup.CoalitionDedupRegistry` tracks which keys already
required backend execution. :func:`~nlp_shap.runtime.dedup.dedup_enabled` maps
``ExplainConfig`` dedup mode to deterministic generation (``auto`` follows
``temperature == 0``).

Hot store sizing
----------------

:class:`~nlp_shap.runtime.store.HotResultStore` is an LRU of coalition key to
generated text. Inside :class:`~nlp_shap.ExplainRunner` it is the scoring buffer
of the run, sized to :math:`2^n` entries so that no output of the run is evicted
before it is scored; the grand coalition is seeded from the base generation.
Memory grows with the number of coalitions actually evaluated, not with
:math:`2^n`.

Scheduling
----------

:class:`~nlp_shap.runtime.scheduler.InferenceScheduler` checks the hot store,
then the dedup registry, then acquires a concurrency semaphore before calling
the backend. :class:`~nlp_shap.runtime.scheduler.SchedulerMetrics` reports
``requested``, ``executed``, ``deduplicated``, ``cache_hits`` and
``kv_cache_hits``.

Failure semantics
~~~~~~~~~~~~~~~~~

The first exception raised by a backend call ends the run: the scheduler cancels
every pending job, collects the exceptions of jobs that failed at the same time
(so none is reported as "never retrieved"), and re-raises the first one. No
metrics and no partial attributions are returned, and an archive written so far
keeps the outputs that completed.

Optional progress
-----------------

Pass a :class:`~nlp_shap.runtime.progress.CoalitionProgress` implementation to
:class:`~nlp_shap.runtime.scheduler.InferenceScheduler` or
:class:`~nlp_shap.ExplainRunner` to receive sync ``on_coalitions_planned`` /
``on_coalition_finished`` callbacks. When omitted,
:class:`~nlp_shap.runtime.progress.NullCoalitionProgress` keeps the hot path
near zero cost.

See also
--------

- :doc:`../guides/runtime` — usage examples
- :doc:`../guides/pipeline` — the stages of a run
- :doc:`../api/runtime` — module reference
- :doc:`../guides/config` — ``archive``, ``dedup`` and ``max_inflight`` settings
- `How a run works
  <https://github.com/Pawlo77/nlp-shap/blob/main/knowledge/handbook/How%20a%20run%20works.md>`_
  in the knowledge vault
