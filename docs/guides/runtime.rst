Runtime core
============

This guide covers the layer under :class:`~mllm_shap.ExplainRunner`: the run
archive, coalition deduplication and the async scheduler. Most users only meet it
through ``explanation.archive``, ``dedup`` and ``max_inflight``; the snippets show
the pieces directly. They share one namespace.

Run archives
------------

An archived run is a directory with ``manifest.json``, the base generation and
one row per coalition output in ``archive.sqlite``. Run an explanation with an
archive path and read the rows back with
:meth:`~mllm_shap.runtime.archive.RunArchive.load`:

.. code-block:: python

   from pathlib import Path

   from mllm_shap import (
       ConversationSnapshot,
       ExplainConfig,
       ExplainRunner,
       Message,
       Role,
       Turn,
   )
   from mllm_shap.runtime import RunArchive

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="Who are you?"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "mock"},
       "explanation": {
           "estimator": "exact",
           "absence_policy": "delete",
           "archive": {"path": "runs/{run_id}"},
       },
   })
   output = ExplainRunner(config).explain_sync(snapshot)
   root = Path("runs") / output.run_id

   print("files:", sorted(path.name for path in root.iterdir()))
   with RunArchive.load(root) as archive:
       print("base:", repr(archive.read_base_generation()[:19]))
       for record in list(archive.history_lazy())[:3]:
           print(record.record_id, repr(record.generation_text[:19]))

.. guide-result:: runtime_archive

Re-running the same configuration on the same snapshot targets the same
directory. :meth:`~mllm_shap.runtime.archive.RunArchive.open` creates archives and
refuses to append to an existing one, so the second run raises
:exc:`~mllm_shap.errors.ArchiveExistsError`; ``on_exists: overwrite`` deletes the
old rows, manifest and base generation first:

.. code-block:: python

   from mllm_shap.errors import ArchiveExistsError

   try:
       ExplainRunner(config).explain_sync(snapshot)
   except ArchiveExistsError as error:
       print(type(error).__name__)

   overwrite = config.model_copy(
       update={
           "explanation": config.explanation.model_copy(
               update={"archive": config.explanation.archive.model_copy(
                   update={"on_exists": "overwrite"},
               )},
           ),
       },
   )
   again = ExplainRunner(overwrite).explain_sync(snapshot)
   print("same run id:", again.run_id == output.run_id)

.. guide-result:: runtime_archive_exists

Run manifest
------------

Every archive stores ``manifest.json`` and one SQLite file, ``archive.sqlite``
(WAL journal), whose rows hold the generated text. The manifest's
``schema_version`` is :data:`~mllm_shap.runtime.archive.MANIFEST_SCHEMA_VERSION`
and the database's ``PRAGMA user_version`` is
:data:`~mllm_shap.runtime.archive.ARCHIVE_SCHEMA_VERSION`.
:meth:`~mllm_shap.runtime.archive.RunArchive.load` refuses archives with any other
version, such as those written by 0.1.17 or earlier, with
:exc:`~mllm_shap.errors.ManifestVersionError`; re-run ``explain`` to rebuild them.
The manifest written by :class:`~mllm_shap.ExplainRunner` records:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``schema_version``, ``mllm_shap_version``
     - Manifest format and the package version that wrote it
   * - ``run_id``, ``snapshot_id``
     - Run identifier and the explained conversation snapshot
   * - ``estimand``, ``estimator``, ``value_fn``, ``normalizer``
     - How payoffs were sampled, scored and aggregated
   * - ``absence_policy``, ``players``, ``player_ids``
     - How the prompt was partitioned and masked; ``player_ids`` follow mask bit
       order
   * - ``backend_kind``, ``model_id``
     - Which backend and model generated the coalition outputs
   * - ``config``
     - Full :class:`~mllm_shap.ExplainConfig` echo

:meth:`~mllm_shap.ExplainRunner.reanalyze` rescores an archive with the runner's
value function and normalizer, re-aggregates with the archived estimator, and
raises :exc:`~mllm_shap.errors.ReanalyzeMismatchError` when the runner's
estimator, absence policy or partition differs from the archived run. With the
same value function it reproduces the ``explain`` attributions; :doc:`pipeline`
has an example.

When every coalition was generated the pipeline marks the archive complete.
:meth:`~mllm_shap.ExplainRunner.reanalyze` raises
:exc:`~mllm_shap.errors.ArchiveIncompleteError` for an archive of a run that failed
part-way, one without the grand or empty coalition, a full-budget archive missing
rows, or an archive whose manifest was not written by
:class:`~mllm_shap.ExplainRunner`.

Build coalition dedup keys
--------------------------

A coalition key hashes everything that determines a generation: snapshot, player
order, mask, absence policy, model and generation settings. With ``dedup.enabled:
auto`` deduplication is on only for deterministic decoding (``temperature`` 0):

.. code-block:: python

   from mllm_shap.pipeline.config import DedupConfig, GenerationConfig
   from mllm_shap.runtime import build_coalition_key, dedup_enabled

   generation = GenerationConfig(temperature=0.0)
   key = build_coalition_key(
       snapshot_id="snap-1",
       player_ids=("snap-1:0:0:0", "snap-1:0:0:1"),
       mask_present=(True, False),
       absence_policy="delete",
       model_id="mock",
       generation=generation,
   )
   print(dedup_enabled(DedupConfig(enabled="auto"), generation), key[:12])

.. guide-result:: runtime_dedup

Schedule async coalition jobs
-----------------------------

:class:`~mllm_shap.runtime.InferenceScheduler` checks the hot store, then the dedup
registry, and only then calls ``generate`` under a ``max_inflight`` semaphore.
Nine jobs over three distinct keys cost three calls:

.. code-block:: python

   import asyncio

   from mllm_shap.masking.codec import MaskCodec
   from mllm_shap.runtime import (
       CoalitionDedupRegistry,
       CoalitionJob,
       HotResultStore,
       InferenceScheduler,
   )

   packed = MaskCodec.pack((True, False, True))

   async def generate(prompt: ConversationSnapshot) -> str:
       return prompt.turns[0].messages[0].text

   scheduler = InferenceScheduler(
       max_inflight=2,
       generation=GenerationConfig(temperature=0.0),
       store=HotResultStore(),
       dedup=CoalitionDedupRegistry(),
   )
   jobs = [
       CoalitionJob(
           coalition_key=f"key-{index % 3}",
           snapshot_id=snapshot.snapshot_id,
           snapshot=snapshot,
           absence_policy="delete",
           mask_words=packed.words,
           mask_n_bits=packed.n_bits,
           model_id="mock",
       )
       for index in range(9)
   ]
   metrics = asyncio.run(scheduler.run(jobs, generate))
   print(metrics)

.. guide-result:: runtime_scheduler

Pass ``progress=`` (any object with ``on_coalitions_planned`` and
``on_coalition_finished``) to observe the counts; :doc:`pipeline` shows one. The
counts are cumulative over every run of one scheduler: when Neyman's second phase
adds jobs, ``on_coalitions_planned`` is called again with the larger total and the
finished count keeps growing.

Streaming large job sets
------------------------

:meth:`~mllm_shap.runtime.scheduler.InferenceScheduler.run_iter` accepts any
iterable and only creates coroutines as slots free up, so an exact run over many
players never holds all :math:`2^n` jobs in memory at once:

.. code-block:: python

   from mllm_shap import ExactEstimator
   from mllm_shap.masking import MaskBuilder, PadPolicy, TokenPartitioner

   players = TokenPartitioner().partition(snapshot)
   builder = MaskBuilder(PadPolicy())

   def job_stream():
       for mask in ExactEstimator.iter_masks(players):
           packed = MaskCodec.pack(mask.present)
           yield CoalitionJob(
               coalition_key=packed.words.hex(),
               snapshot_id=snapshot.snapshot_id,
               snapshot=builder.render(builder.view(snapshot, players, mask)),
               absence_policy="pad",
               mask_words=packed.words,
               mask_n_bits=packed.n_bits,
               model_id="mock",
           )

   streaming = InferenceScheduler(
       max_inflight=4,
       generation=GenerationConfig(temperature=0.0),
       store=HotResultStore(),
   )
   metrics = asyncio.run(streaming.run_iter(job_stream(), generate, total=8))
   print(metrics.requested, metrics.executed)

.. guide-result:: runtime_run_iter

Failure semantics
-----------------

Jobs run in one :class:`asyncio.TaskGroup`. When any ``generate`` call raises,
the task group cancels every other job and the scheduler re-raises that first
exception itself, not wrapped in an :class:`ExceptionGroup`, so ``except
RuntimeError`` works as written; a run never returns partial metrics or a
partially aggregated result:

.. code-block:: python

   async def flaky(prompt: ConversationSnapshot) -> str:
       if "[MASK]" in prompt.turns[0].messages[0].text:
           msg = "backend refused a masked prompt"
           raise RuntimeError(msg)
       return "ok"

   failing = InferenceScheduler(
       max_inflight=2,
       generation=GenerationConfig(temperature=0.0),
       store=HotResultStore(),
   )
   try:
       asyncio.run(failing.run_iter(job_stream(), flaky, total=8))
   except RuntimeError as error:
       print("raised:", error)

.. guide-result:: runtime_failure

Logging
-------

Importing ``mllm_shap`` leaves the host application's logging alone. Opt in with
:func:`~mllm_shap.configure_logging`:

.. code-block:: python

   import mllm_shap

   mllm_shap.configure_logging(level="INFO")              # plain lines on stderr
   mllm_shap.configure_logging(level="INFO", json=True)   # one JSON object per line
   mllm_shap.configure_logging("config/logging.example.toml")  # a [tool.logging] file

Without arguments it applies the ``[tool.logging]`` section of the nearest
``pyproject.toml``. Setting ``MLLM_SHAP_LOGGING_CONFIG`` to a TOML file applies that
file on import instead; ``config/logging.example.toml`` in the repository is a
starting point. At ``INFO`` the logger ``mllm_shap.pipeline.orchestrator`` reports
each stage once:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Event
     - Message
   * - Run start
     - ``explain run <id> started: <n> players, estimator <name>, backend <kind>``
   * - Base generation
     - ``base generated in <ms> ms``
   * - Coalitions scheduled
     - ``<count> coalitions scheduled`` (once per sampling phase)
   * - Run end
     - ``explain run <id> done in <ms> ms (<n> coalitions, <calls> backend calls)``

``DEBUG`` adds HTTP retries and skipped plugin entry points; LM Studio model
downloads are logged at ``INFO`` by ``mllm_shap.backends.lmstudio.models``.

Telemetry and progress
----------------------

:class:`~mllm_shap.runtime.telemetry.InMemoryObservabilitySink` records one
:class:`~mllm_shap.runtime.telemetry.SpanRecord` (name and duration) per pipeline
stage, and any :class:`~mllm_shap.runtime.progress.CoalitionProgress` receives the
coalition counts:

.. code-block:: python

   from mllm_shap import InMemoryObservabilitySink


   class CountingProgress:
       def __init__(self) -> None:
           self.planned: list[int] = []
           self.finished = 0

       def on_coalitions_planned(self, total: int) -> None:
           self.planned.append(total)

       def on_coalition_finished(self, done: int, total: int) -> None:
           self.finished = done


   sink = InMemoryObservabilitySink()
   progress = CountingProgress()
   ExplainRunner(overwrite, telemetry=sink, progress=progress).explain_sync(snapshot)
   print("stages:", [span.name for span in sink.spans])
   print("planned:", progress.planned, "finished:", progress.finished)

.. guide-result:: runtime_telemetry

A progress bar needs nothing more. With ``tqdm`` installed:

.. code-block:: python

   from tqdm import tqdm


   class TqdmProgress:
       def __init__(self) -> None:
           self.bar = tqdm(total=0, unit="coalition")

       def on_coalitions_planned(self, total: int) -> None:
           self.bar.total = total
           self.bar.refresh()

       def on_coalition_finished(self, done: int, total: int) -> None:
           self.bar.update(done - self.bar.n)


   ExplainRunner(config, progress=TqdmProgress()).explain_sync(snapshot)

The callbacks run synchronously on the event loop, so keep them cheap.

Performance benchmarks
----------------------

Benchmarks live in ``tests/benchmarks/`` and use ``pytest-benchmark``. They cover
coalition deduplication keys, mask pack/unpack, 10k-row archive round trips,
prefix-cache lookups, exact, stratified and complementary aggregation, the
scheduler with 1 000 jobs, TF-IDF scoring and an end-to-end mock run.

.. code-block:: bash

   make bench              # run them and write .pytest_benchmark/benchmark.json
   make bench-regression   # fail when a median is >25% slower than baseline.json

``tests/benchmarks/baseline.json`` is recorded on the maintainer's machine (named
in ``tests/benchmarks/AGENTS.md``), so compare against it on comparable hardware.
``make check`` does not run benchmarks; a weekly CI workflow runs ``make bench``
and keeps the report as an artifact. Run ``make bench-regression`` before changing
hot paths in ``runtime/``, ``masking/`` or ``estimation/``.

Notebook
--------

For a runnable walkthrough with stored outputs, see
``examples/runtime_core.ipynb`` (also embedded on :doc:`../examples`).
