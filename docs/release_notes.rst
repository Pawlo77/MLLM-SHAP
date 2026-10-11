Release notes
=============

Changes in each published ``mllm-shap`` version. Newest first.

.. _release-unreleased:

Unreleased
----------

.. _release-0-4-1:

0.4.1 (2026-10-11)
------------------

Added
~~~~~

- The ``transformers`` backend hears audio: a checkpoint ``AutoModelForCausalLM``
  does not know (or any ``backend.options.model_class``) loads with its processor
  and takes audio parts, for example Granite Speech (:doc:`guides/audio_backends`).
  ``backend.options.attn_implementation`` is passed to ``from_pretrained``.
- The ``api`` backend sends audio parts as OpenAI ``input_audio`` parts
  (base64 WAV) instead of refusing them.
- The ``lmstudio`` option ``audio: transcript`` sends the text parts of a message
  with audio; by default audio is still refused, since the LM Studio SDK takes none.

Changed
~~~~~~~

- The ``transformers`` extra adds ``torchaudio`` (Granite Speech's feature
  extractor needs it), and the backend reads the model config before loading.

.. _release-0-4-0:

0.4.0 (2026-10-11)
------------------

Added
~~~~~

- The ``mock_audio`` backend (:class:`~mllm_shap.backends.mock.audio.MockAudioBackend`)
  echoes message words and answers one token per voiced run of each audio part,
  so audio players get attributions in tests and demos without a model. It reads
  16-bit WAV without the ``audio`` extra.

.. _release-0-3-1:

0.3.1 (2026-10-11)
------------------

Fixed
~~~~~

- The ``api`` backend gives an empty reply the logprobs ``()`` instead of ``None``
  (LM Studio sends ``logprobs: null`` for it), so ``reanalyze`` accepts archives of
  runs that contain empty generations.

.. _release-0-3-0:

0.3.0 (2026-10-11)
------------------

Added
~~~~~

- ``players: tokens`` takes the plugin option ``roles`` (like ``messages``): only
  the words of the listed roles become players, other messages stay present.
- :func:`~mllm_shap.viz.plot_conversation` and
  :func:`~mllm_shap.viz.render_conversation_html` draw a conversation message by
  message under role and speaker headers, with token or message players shaded in
  place and text that is no player in grey. Messages with audio players get a
  waveform strip shaded on the same colour scale.
- The ``sgpa_segments`` option ``text_roles`` also makes the words of typed
  messages of those roles players, so typed and spoken words share one run.

Changed
~~~~~~~

- The project is renamed to ``mllm-shap``: install ``mllm-shap`` from PyPI,
  import ``mllm_shap``, and find the repository at ``Pawlo77/MLLM-SHAP``.
  Environment variables start with ``MLLM_SHAP_``, entry-point groups with
  ``mllm_shap.``, and the base error is ``MllmShapError``. The old
  ``nlp-shap`` package is archived.
- :func:`~mllm_shap.viz.plot_audio_segments` with ``values`` lightens every second
  segment and keeps fills translucent, so neighbouring words stay distinct (equal
  values no longer merge into one block) and the waveform shows through.

.. _release-0-2-1:

0.2.1 (2026-10-10)
------------------

Added
~~~~~

- ``explanation.players: sgpa_segments`` (:class:`~mllm_shap.SgpaSegmentPartitioner`)
  makes each spoken word of an audio message's transcript a player. Its
  :class:`~mllm_shap.PlayerSpan` carries the new ``samples`` range, and
  ``delete`` cuts absent audio while ``pad`` and ``neutral`` silence it.
  See :doc:`guides/alignment`.
- :meth:`mllm_shap.Message.with_audio` replaces a message's first audio part;
  :class:`~mllm_shap.alignment.AudioSegment` records ``token_index`` and the rough
  CTC edges ``ctc_start_time`` and ``ctc_end_time``.
- :func:`~mllm_shap.viz.plot_audio_segments` draws aligned segments over the
  waveform, the shift of each boundary from its CTC edge and the RMS energy;
  ``values`` colours segments by attribution.

.. _release-0-2-0:

0.2.0 (2026-10-10)
------------------

Added
~~~~~

- :mod:`mllm_shap.alignment` (``audio`` extra): :class:`~mllm_shap.alignment.SpectrogramGuidedAligner`
  aligns a transcript to its speech in an :class:`~mllm_shap.AudioPart` and returns one
  :class:`~mllm_shap.alignment.AudioSegment` per token. Each boundary moves to a quiet
  point between words. :class:`~mllm_shap.alignment.Wav2Vec2Ctc` runs a Hugging Face
  Wav2Vec2 model (``transformers`` extra); any :class:`~mllm_shap.alignment.CtcModel`
  can replace it. See :doc:`guides/alignment`.

.. _release-0-1-19:

0.1.19 (2026-10-09)
-------------------

Fixed
~~~~~

- The transformers backend warns once when ``explanation.kv_cache`` runs with
  ``float16`` or ``bfloat16`` weights: cached prefixes round differently from one
  forward pass, so greedy outputs can differ from the cache off. The backends guide
  no longer promises equality; ``backend.options.dtype: float32`` is identical.
- The LM Studio backend accepts a model key that LM Studio lists once per
  downloaded quantisation; earlier versions refused it as ambiguous.

.. _release-0-1-18:

0.1.18 (2026-10-09)
-------------------

Fixed
~~~~~

- Without model embeddings (mock, LM Studio and API backends),
  ``embedding_mode: contextual`` now hashes the generated text alone, so
  ``explain`` and ``reanalyze`` agree and ``reanalyze --strict`` accepts the
  archive (``embedding_source: hash``). The values of such runs change.
- The ``token_text`` and ``token_bar`` renderers no longer log matplotlib's
  ``Failed to find font weight semibold`` warning.
- Renderer labels no longer overlap: ``token_text`` wraps long text onto new
  lines, and ``token_bar`` puts value labels beyond the bar ends and the sign
  legend outside the axes.
- Complementary and Neyman samplers now evaluate all ``n`` singleton pairs
  ``({i}, N - {i})`` after the grand and empty pair. On an additive 8-player game
  at ``budget.fraction: 0.5`` the attribution sum was on average 1.7 above
  ``grand_value - base_value``; it is now within 0.03.
- Neyman phase one walks its grid size by size, round-robin over players, so a
  short budget leaves the same sizes empty for every player; its small-budget
  error now matches complementary sampling (:doc:`theory/approximation`).
- Complementary and Neyman budgets never drop below ``2n + 2`` masks (``2 ** n``
  when smaller), so all singleton pairs fit. With the earlier ``2n`` floor, the
  game ``v(S) = |S|`` with three players gave ``[1, 1, 1.333]`` instead of
  ``[1, 1, 1]``.

- Monte Carlo and other partial-sample estimates log a ``WARNING`` naming the
  players with no observed marginal (their value is 0), typically at small
  budgets where only the grand and empty coalitions were sampled.
- The transformers backend reuses cached prompt-prefix KV state (every prompt
  token but the last). A miss computes the prefix once and stores it. A hit
  copies the longest cached prefix, crops it to the shared tokens and extends it.
  No prompt token is processed twice; earlier versions ran one forward pass per
  prefix length.
- ``kv_cache.max_entries`` (default 8) bounds the prefix states, and model calls
  (tokenization included) are serialized. ``metrics.kv_cache_reused_tokens``
  reports the prompt tokens served from the cache next to
  ``metrics.kv_cache_hits``. Both count the current run only, also when
  ``ExplainRunner(backend=...)`` reuses the backend. In ``float32`` greedy outputs
  are identical with the cache on and off.
- The transformers backend runs on Apple MPS when available. The new
  ``backend.options.dtype`` (``auto``, ``float32``, ``float16``, ``bfloat16``)
  defaults to ``auto``: bfloat16 on CUDA and MPS when supported, float16 on other
  accelerators and float32 on CPU. Earlier versions loaded float16 on every
  accelerator, which overflows to NaN on many bfloat16-trained models.
- ``backend.options.revision`` pins the Hub revision of the model and tokenizer.
- The transformers backend computes static and contextual embeddings only when
  the value function needs them (see
  :class:`~mllm_shap.protocols.value.RequiresEmbeddings`), and ``explain`` then
  scores the model's embeddings. Earlier versions ran two extra forward passes
  per coalition and discarded the result.
- :meth:`~mllm_shap.ExplainRunner.reanalyze` still scores hash embeddings of the
  archived text, because archives store text only. Embedding-based values of a
  transformers run therefore differ
  between ``explain`` and ``reanalyze``.
- The manifest records which embeddings the run scored: ``embedding_source``
  (``none``, ``hash`` or ``backend``) and ``embedding_model``. Reanalyzing a
  ``backend`` run with an embedding value function logs a ``WARNING``;
  ``ExplainRunner.reanalyze(..., strict=True)`` (``mllm-shap reanalyze --strict``)
  raises :exc:`~mllm_shap.errors.ReanalyzeMismatchError` instead.
- The HTTP API backend retries status 408, 429, 500, 502, 503 and 504 and
  transport errors up to ``backend.max_retries`` times. The backoff is
  exponential and honours ``Retry-After`` up to ``backend.timeout_s``. Non-JSON
  responses raise :exc:`~mllm_shap.errors.BackendUnavailableError`, and at most
  1024 responses are cached.
- The HTTP API backend sends ``MLLM_SHAP_API_KEY`` as the bearer key to any host.
  It sends ``OPENAI_API_KEY`` only when that is unset and ``backend.api_host`` is
  ``api.openai.com``; earlier versions sent it to every host, local servers
  included.
- The LM Studio backend retries only connection failures and re-raises
  ``ValueError`` and ``TypeError`` at once. It matches downloaded models exactly
  first, then by the prefix of the last path segment, then by substring; an
  ambiguous id raises with the list of candidates.
- With ``auto_download``, the LM Studio backend refuses
  (:exc:`~mllm_shap.errors.BackendUnavailableError`, listing the hits) when no
  repository hit is flagged exact or matches the name. Earlier versions
  downloaded the first hit.
- :meth:`~mllm_shap.ExplainRunner.explain_sync` called inside a running event
  loop raises a ``RuntimeError`` that points to ``await runner.explain(...)``.
  With an injected backend, repeated ``explain_sync`` calls run on one event loop
  owned by the runner, so loop-bound clients and locks keep working.
- :meth:`~mllm_shap.ExplainRunner.reanalyze` refuses with
  :exc:`~mllm_shap.errors.ArchiveIncompleteError` the archive of a run that failed
  part-way, even before its first coalition. It also refuses archives without
  the grand or empty coalition, full-budget archives (``budget.fraction: 1.0``)
  missing rows, and archives not written by :class:`~mllm_shap.ExplainRunner`.
- Re-running explain replaces the archive of a run that did not finish without
  needing ``on_exists: overwrite``.
- :meth:`~mllm_shap.runtime.archive.RunArchive.load` opens the database read-only
  and writes nothing, so archives on read-only storage can be reanalyzed. It
  raises :exc:`~mllm_shap.errors.ArchiveNotFoundError` (a ``FileNotFoundError``)
  when ``manifest.json`` or ``archive.sqlite`` is missing, instead of creating an
  empty database.
- :meth:`~mllm_shap.runtime.archive.RunArchive.load` raises
  :exc:`~mllm_shap.errors.ArchiveCorruptError` when either file cannot be read or
  the manifest's fields are invalid, so the command line exits with 2 on a
  damaged archive.
- Monte Carlo sampling counts the grand and empty coalitions toward the budget;
  earlier versions could exceed it by two calls.
- Progress callbacks report one growing plan and a monotonic finished count
  across the two Neyman phases.
- Partial-sample marginal aggregation renormalizes over observed strata with the
  semivalue's stratum probabilities, for any estimand.
- ``explanation.neyman`` and ``explanation.embedding_mode`` now reach the
  Neyman estimator and the embedding value functions; earlier versions ignored
  them. ``neyman.initial_fraction`` must lie in ``(0, 1]`` and
  ``neyman.initial_num_samples`` be at least 1.
- ``run_id`` hashes the model id of an injected backend, not the one in
  ``backend.model_id``.

Changed
~~~~~~~

- The ``api`` backend requires ``backend.api_host``; the ``OPENAI_BASE_URL``
  environment fallback and the ``http://127.0.0.1:1234/v1`` default are gone, so
  a configuration without a host fails at load time with :class:`~mllm_shap.ConfigError`
  instead of silently targeting a local server.
- The coalition scheduler runs jobs in an :class:`asyncio.TaskGroup`. The first
  failing job still cancels the others and its exception is raised unwrapped,
  not as an ``ExceptionGroup``, also when it sits next to a nested group of
  cancellations.
- ``estimator: complementary`` and ``neyman_cc`` with ``estimand: banzhaf`` are
  refused with :exc:`~mllm_shap.errors.ConfigError` when the config is validated
  instead of failing after every coalition was generated.
- The ``context_id`` argument of
  :func:`~mllm_shap.value.generation_records.generation_record_from_text` is
  ignored and raises a ``DeprecationWarning``; it will be removed in 0.2.0.
  Custom backends call ``generation_record_from_text(text)``.

- Plugin factories receive a :class:`~mllm_shap.plugins.PluginContext` (the run's
  config, the plugin's options and the registry), and every built-in plugin
  class takes it as its first constructor argument.
  :meth:`~mllm_shap.plugins.PluginRegistry.resolve` takes the context as its
  third argument. Third-party plugins must accept ``context``.
- LM Studio settings ``hub_search``, ``quantization``, ``auto_download`` and
  ``serialize_generate`` moved from ``backend`` to ``backend.options`` and are
  validated by the LM Studio backend; unknown keys raise
  :exc:`~mllm_shap.errors.ConfigError`.
- Backends take a :class:`~mllm_shap.plugins.PluginContext` instead of a
  ``BackendConfig``, and ``generate(snapshot, generation)`` receives the whole
  :class:`~mllm_shap.domain.config.GenerationConfig`.
- :class:`~mllm_shap.domain.conversation.Message` holds ``content`` parts
  (:class:`~mllm_shap.domain.conversation.TextPart`,
  :class:`~mllm_shap.domain.conversation.AudioPart`); ``Message(role=...,
  text=...)`` still works and ``Message.text`` joins the text parts. Snapshot ids
  of text-only conversations are unchanged; audio is detected by part type, no
  longer by an ``audio:`` text prefix.
- Partitioners return a :class:`~mllm_shap.domain.players.PlayerSet` with one
  :class:`~mllm_shap.domain.players.PlayerSpan` per player. Absence policies and
  labels read the spans instead of re-tokenising, so custom partitioners work in
  :class:`~mllm_shap.ExplainRunner`, and text outside every span is kept. Rendered
  messages keep their original whitespace instead of collapsing it to single
  spaces.
- ``GenerationConfig``, ``DedupConfig`` and ``KvCacheConfig`` live in
  :mod:`mllm_shap.domain.config` (still importable from
  ``mllm_shap.pipeline.config``); the runtime no longer imports the pipeline.
- ``import mllm_shap`` no longer reads ``[tool.logging]`` from the nearest
  ``pyproject.toml``. Set ``MLLM_SHAP_LOGGING_CONFIG`` or call
  :func:`mllm_shap.configure_logging`.
- The orchestrator logs run start, base generation, scheduled coalitions and
  run duration at ``INFO``.
- ``kv_cache.enabled`` defaults to ``false``; new ``kv_cache.max_entries``.
- New ``backend.timeout_s`` (default 120) and ``backend.max_retries`` (default
  3); the HTTP client caps connections at ``explanation.max_inflight``.
- Run archives store generation text in the SQLite row (no ``blobs/``
  directory), use WAL journaling and carry ``user_version`` 3 and manifest
  schema version 3. Archives written by 0.1.17 raise
  :exc:`~mllm_shap.errors.ManifestVersionError`; re-run explain to rebuild them.
- Coalition jobs are rendered lazily while the scheduler consumes them.
- :func:`~mllm_shap.explain_config_from_yaml` raises
  :exc:`~mllm_shap.errors.ConfigError` (not pydantic's ``ValidationError``) for
  settings that contradict each other, and for an empty document or a YAML root
  that is not a mapping (earlier a ``TypeError``).
- The ``transformers`` extra requires ``transformers>=4.56``; the build backend
  is ``uv_build`` 0.12.

Added
~~~~~

- ``Message(speaker=...)`` names who wrote a message when people share a role.
  It is part of the snapshot id and of snapshot JSON (``"speaker"``), backends
  send ``"<speaker>: <text>"``, and labels and ``to_dict`` report it.
- ``explanation.players: messages`` (:class:`~mllm_shap.masking.MessagePartitioner`)
  makes each message one player labelled ``"<role>[/<speaker>] <n>: <text>"``;
  plugin option ``roles`` keeps other roles always present.
- ``value_fn: logprob`` scores the mean log probability of each coalition's own
  generated tokens. The transformers and ``api`` backends return logprobs
  (:class:`~mllm_shap.protocols.backend.SupportsLogprobs`); other backends raise
  :exc:`~mllm_shap.errors.ConfigError`. Archives store them (``user_version`` and
  manifest schema 4), so older archives need a new ``explain``.
- ``mllm-shap`` command line (:doc:`guides/cli`): ``explain`` (JSON result,
  optional PNG chart), ``reanalyze`` (``--strict`` refuses runs scored with
  backend embeddings), ``plugins`` and ``config --example`` / ``--check``.
- The command line uses ``argparse`` only. It exits with 2 for invalid input
  (arguments, ``--log-level``, configuration, snapshot, unreadable input files or
  archives) and with 3 when the backend is unavailable or the connection fails.
  Any other exception propagates with its traceback.
- :meth:`mllm_shap.ConversationSnapshot.from_json` and
  :meth:`mllm_shap.ConversationSnapshot.to_json` read and write snapshots as
  ``{"turns": [{"messages": [{"role": ..., "text": ...}]}]}``, with an optional
  ``"snapshot_id"`` when the id is not the content-derived one, so the id (and
  the ``run_id``) survives a round trip. Messages with several content parts are
  refused, like audio.
- :meth:`mllm_shap.ExplainRunOutput.to_dict` returns a JSON-compatible result
  with a stable key order, including one label per player (text and character
  span); :attr:`ExplainRunOutput.player_set
  <mllm_shap.ExplainRunOutput.player_set>` holds the explained players as a
  :class:`~mllm_shap.PlayerSet`.
- :func:`~mllm_shap.configure_logging` accepts ``level=`` and ``json=True`` to log
  the pipeline's stage events to stderr without a TOML file. Events are plain
  lines or one JSON object per line, and the package logger then stops
  propagating to the root logger. ``json=True`` with a ``config_path`` raises
  ``ValueError``.
  ``config/logging.example.toml`` is a starting point for
  ``MLLM_SHAP_LOGGING_CONFIG``.
- ``explanation.budget.evaluations`` sets an absolute budget instead of
  ``budget.fraction``: that many coalition evaluations in total, the empty and
  grand coalitions included, at any number of players.
- Each estimator applies its floor to ``budget.evaluations`` and the ceiling
  ``2 ** n``: 2 for ``mc``, ``2n + 2`` for ``complementary`` and ``neyman_cc``
  (which also round an odd count down to whole pairs). ``exact`` accepts only
  ``fraction: 1.0``, and setting both keys raises
  :exc:`~mllm_shap.errors.ConfigError`.
- ``max_players`` refuses ``budget.evaluations`` above ``2 ** max_players``.
  Configurations without ``evaluations`` serialise as before, so their run ids do
  not change.

- ``explanation.plugin_options``: per-plugin options keyed by
  ``<group>/<name>`` (for example ``absence_policies/pad: {placeholder:
  "<unk>"}``) or by the bare plugin name when only one plugin of the run has it.
  Built-in plugins declare their options in a
  :class:`~mllm_shap.protocols.plugin.PluginOptions` model; unknown keys, wrong
  types and keys that name no plugin of the run raise
  :exc:`~mllm_shap.errors.ConfigError` before the run starts, naming the group,
  plugin and key.
- Error hierarchy rooted at :exc:`~mllm_shap.errors.MllmShapError`:
  :exc:`~mllm_shap.errors.PluginNotFoundError`,
  :exc:`~mllm_shap.errors.PluginConflictError`,
  :exc:`~mllm_shap.errors.PluginLoadError`, :exc:`~mllm_shap.errors.ConfigError`,
  :exc:`~mllm_shap.errors.ArchiveNotFoundError` and
  :exc:`~mllm_shap.errors.ArchiveCorruptError` join the existing errors.
- Config cross-validation: ``estimator: exact`` requires ``budget.fraction: 1.0``
  and ``neyman_cc`` rejects ``include_minimal_masks: true``.
- Capability protocols :class:`~mllm_shap.protocols.SupportsKvCache`,
  :class:`~mllm_shap.protocols.SupportsAclose`,
  :class:`~mllm_shap.protocols.FittableValueFunction` and
  :class:`~mllm_shap.protocols.value.RequiresEmbeddings`; any value function with
  ``fit`` is fitted on the run corpus, not only TF-IDF.
  ``GenerationConfig.compute_embeddings`` asks a backend for model embeddings.
- :func:`~mllm_shap.plugins.default_registry` registers all eight plugin groups
  (estimands and renderers included) and loads third-party entry points;
  registering a name twice raises :exc:`~mllm_shap.errors.PluginConflictError`
  unless ``replace=True``. :attr:`~mllm_shap.ExplainRunner.registry` can be passed
  to :func:`~mllm_shap.render_attribution` to reuse custom renderers.
- Top-level exports for every built-in plugin class, the absence policies,
  :class:`~mllm_shap.TokenPartitioner`, all errors, the observability and
  progress protocols and ``display_attribution_html``.
- ``ExplainRunner(backend=...)`` reuses one backend across ``explain`` calls;
  :meth:`~mllm_shap.ExplainRunner.close`, :meth:`~mllm_shap.ExplainRunner.aclose`
  and ``with`` / ``async with`` blocks close it.
- New ``explanation.max_players`` (default 24): a run whose budget plans more
  than ``2 ** max_players`` coalitions (``budget.fraction * 2 ** n`` for ``n``
  players, whatever the estimator) raises :exc:`~mllm_shap.errors.ConfigError`.

Removed
~~~~~~~

- ``generation.precompute_base`` (the base generation always runs; remove the key
  from YAML configs), ``CoalitionJob.utility``, ``CoalitionRecord.utility``,
  ``SystemRolesSetup``, ``CooperativeGame.empty_value`` and
  ``masking.prefix_tree.PrefixTreeStub``.

Documentation
~~~~~~~~~~~~~

- New "How a run works" page (stages, scheduler counters, archives,
  ``reanalyze``, progress and telemetry, ``explain`` vs ``explain_sync``),
  "Choosing an estimator", and "Coming from MLLM-Shap-legacy" guides; citing and
  contributing pages.
- Configuration reference and the public-exports table are generated from the
  installed package; every runnable guide snippet is executed on each docs
  build from the page source.
- Theory pages state the efficiency convention
  ``sum(values) = grand_value - base_value`` (exact on the full coalition table
  only), the stratified partial-sample estimand, and corrected DOIs
  (Štrumbelj & Kononenko 2014, Young 1985).
- ``make docs`` builds with ``-W --keep-going -n``; ``make -C docs linkcheck``
  checks external links.

Tooling
~~~~~~~

- Delivery autopilot: skill ``ship`` opens the pull request with a Copilot review
  request, addresses the threads and merges routine changes (``scripts/pr_policy.py``)
  itself; other changes stop before the merge for the maintainer.
- Upgrades and releases: ``make upgrade-check`` and ``make upgrade``
  (``scripts/upgrade.py``) list and apply new package versions, prek hook revisions,
  workflow action pins, the uv version and Obsidian plugins, except the pins in
  ``publish.yaml``. ``scripts/pr_policy.py --bump`` derives the next version from the
  commits since the last tag; skill ``release`` commits it and tags after the merge.
- Benchmarks use ``pytest-benchmark``: ``make bench`` writes a JSON report and
  ``make bench-regression`` fails when a median is more than 25% slower than the
  committed ``tests/benchmarks/baseline.json``. New benchmarks cover exact,
  stratified and complementary aggregation, the scheduler, TF-IDF scoring and an
  end-to-end mock run.
- ``make mutate`` runs ``mutmut`` on the estimation package.
- Tests that build a wheel or reach the network are marked ``slow`` and run in
  ``make tests-all`` only.
- Practical tests: ``make practical FEATURE=<id|glob|all>`` runs about 30
  scenarios (marker ``practical``) on ``Qwen/Qwen2.5-1.5B-Instruct`` and
  ``SmolLM2-360M-Instruct`` on a GPU, with rubric checks and a logit judge. Skill
  ``ship`` runs the scenarios ``scripts/practical_features.py`` maps from the
  diff; ``make check`` excludes them.
- Deep practical tier: ``make practical TIER=deep`` runs nine ``deep.*``
  scenarios on ``Qwen/Qwen2.5-3B-Instruct`` under an 8 GB VRAM cap. They cover error
  versus budget against a 10-player exact reference, estimator stability, Banzhaf, a
  counterfactual pair per value function, a 10-turn three-speaker thread, the KV
  cache, archive reanalysis and CLI parity. Skill ``release`` runs it.

.. _release-0-1-17:

0.1.17 (2026-10-07)
-------------------

Fixed
~~~~~

- Exact, Monte Carlo, complementary and Neyman attributions now satisfy
  efficiency on the full coalition table (``budget.fraction: 1.0``); approximate
  budgets converge to it as coverage grows. Earlier versions omitted the grand coalition (exact, Monte Carlo)
  and the :math:`(N, \emptyset)` pair (complementary, Neyman), which
  under-reported every player by its :math:`S = N\setminus\{i\}` marginal.
  Re-run explanations produced by 0.1.x.
- Coalition payoffs stay aligned with their masks when ``kv_cache.enabled``
  reorders jobs for prefix reuse; earlier versions could score a coalition with
  another coalition's generation.
- Monte Carlo partial samples use a stratified mean of observed marginals
  instead of an un-normalised weighted sum.
- Complementary and Neyman aggregation pair masks by complement, independent of
  evaluation order.
- Neyman phase one no longer stalls on cells that cannot be filled, and phase two
  now runs inside :class:`~mllm_shap.ExplainRunner` and spends the remaining
  budget. The allocation uses the complement column :math:`n - j`.
- :class:`~mllm_shap.runtime.scheduler.InferenceScheduler` surfaces the first
  coalition failure and cancels the remaining jobs.

Changed
~~~~~~~

- ``run_id`` includes a digest of the configuration and model id. Re-running into
  an existing archive raises :exc:`~mllm_shap.errors.ArchiveExistsError` unless
  ``explanation.archive.on_exists`` is ``overwrite``.
- Manifest schema version 2 records the estimator, value function, normalizer,
  absence policy, partition, backend, model, player ids and the full config.
  Archives also carry SQLite ``user_version`` 2; older archives raise
  :exc:`~mllm_shap.errors.ManifestVersionError` on load.
- :meth:`~mllm_shap.ExplainRunner.reanalyze` re-aggregates with the archived
  estimator, refuses a mismatched estimator, absence policy or partition with
  :exc:`~mllm_shap.errors.ReanalyzeMismatchError`, and reproduces ``explain``
  values for the same value function.
- Value functions score records built from the generated text in both
  ``explain`` and ``reanalyze``
  (:func:`~mllm_shap.value.generation_records.generation_record_from_text`).
- :meth:`~mllm_shap.masking.codec.MaskCodec.hash_mask` is stable across processes.
- Complementary and Neyman estimators raise ``ValueError`` for the ``banzhaf``
  estimand; earlier versions returned complementary Shapley values under a
  Banzhaf label.

Added
~~~~~

- :attr:`~mllm_shap.pipeline.result.ExplainResult.base_value` and
  :attr:`~mllm_shap.pipeline.result.ExplainResult.grand_value`.
- :class:`~mllm_shap.protocols.estimator.TwoPhaseEstimator` protocol; every
  estimator exposes ``estimate_attributions(masks, payoffs, aggregator)``.

Removed
~~~~~~~

- ``InferenceScheduler.run_stream`` (use
  :meth:`~mllm_shap.runtime.scheduler.InferenceScheduler.run_iter`) and the
  ``mllm_shap.backends.mock.generation`` module.

.. _release-earlier:

Earlier releases
----------------

- **0.1.16** (2026-07-19): coalition progress callbacks on the scheduler and
  runner; ``BackendConfig.serialize_generate`` for LM Studio; ``DeletePolicy``
  renders the empty coalition; a Neyman infinite loop at small player counts is
  fixed.
- **0.1.15** (2026-07-15): runtime micro-benchmarks and the ``make bench`` and
  ``make bench-regression`` targets.
- **0.1.14** (2026-07-15): :doc:`guides/extending` guide; renderers load lazily,
  so ``import mllm_shap`` works without ``[viz]``.
- **0.1.13** (2026-07-15): :mod:`mllm_shap.viz` with the ``token_text`` and
  ``token_bar`` renderers and the ``[viz]`` extra (on PyPI; no ``v0.1.13`` git
  tag).
- **0.1.12** (2026-07-15): :class:`~mllm_shap.backends.api.ApiBackend` for
  OpenAI-compatible HTTP APIs and the ``[api]`` extra.
- **0.1.11** (2026-07-15):
  :class:`~mllm_shap.backends.transformers.text.TransformersTextBackend`, prefix
  KV caching and the ``[transformers]`` extra.
- **0.1.10** (2026-07-15): :class:`~mllm_shap.backends.lmstudio.LmStudioBackend`
  and the ``[lmstudio]`` extra.
- **0.1.9** (2026-07-15): :meth:`~mllm_shap.pipeline.runner.ExplainRunner.reanalyze`,
  stage telemetry and :class:`~mllm_shap.runtime.metrics.PerfSummary`.
- **0.1.8** (2026-07-14): :class:`~mllm_shap.pipeline.runner.ExplainRunner`, the
  orchestrator and the ``mock`` backend.
- **0.1.7** (2026-07-14): TF-IDF, embedding and logprob value functions,
  presentation normalizers and :class:`~mllm_shap.GenerationRecord`.
- **0.1.6** (2026-07-14): the ``mc``, ``complementary`` and ``neyman_cc``
  estimators.
- **0.1.5** (2026-07-13): :class:`~mllm_shap.estimation.exact.ExactEstimator`
  and :meth:`~mllm_shap.runtime.scheduler.InferenceScheduler.run_iter`.

.. _release-0-1-4:

0.1.4 (2026-07-13)
------------------

Added
~~~~~

- :class:`~mllm_shap.runtime.archive.RunArchive` with SQLite index, blob storage,
  and lazy :meth:`~mllm_shap.runtime.archive.RunArchive.history_lazy` iteration.
- :class:`~mllm_shap.runtime.archive.CoalitionRecord` rows with mask bytes,
  utility, timing, and cache-hit metadata.
- :class:`~mllm_shap.runtime.dedup.CoalitionDedupRegistry` and
  :func:`~mllm_shap.runtime.dedup.build_coalition_key` SHA256 coalition keys.
- :class:`~mllm_shap.runtime.store.HotResultStore` LRU cache for recent generations.
- :class:`~mllm_shap.runtime.scheduler.InferenceScheduler` with bounded
  ``max_inflight`` concurrency and :class:`~mllm_shap.runtime.scheduler.SchedulerMetrics`.

Documentation
~~~~~~~~~~~~~

- Theory and usage guides for the runtime archive, dedup, and scheduler.
- API reference page for :mod:`mllm_shap.runtime`.
- Example notebook ``examples/runtime_core.ipynb``.

.. _release-0-1-3:

0.1.3 (2026-07-12)
------------------

Added
~~~~~

- Masking codec: :class:`~mllm_shap.masking.codec.PackedMask`,
  :class:`~mllm_shap.masking.codec.MaskCodec` with legacy-compatible packed hashing.
- :class:`~mllm_shap.masking.space.MaskSpace` for projecting feature splits to full masks.
- View-based :class:`~mllm_shap.masking.builder.MaskedSnapshot` and
  :class:`~mllm_shap.masking.builder.MaskBuilder` with structural sharing.
- Absence policies: :class:`~mllm_shap.masking.policies.DeletePolicy`,
  :class:`~mllm_shap.masking.policies.PadPolicy`,
  :class:`~mllm_shap.masking.policies.NeutralPolicy`.
- :class:`~mllm_shap.masking.partitions.TokenPartitioner` and built-in plugin
  registration for ``tokens``, ``delete``, ``pad``, and ``neutral``.
- ``numpy`` core dependency for mask packing.

Documentation
~~~~~~~~~~~~~

- Theory and usage guides for coalition masking and absence policies.
- API reference page for :mod:`mllm_shap.masking`.
- Example notebook ``examples/masking_views.ipynb``.

.. _release-0-1-2:

0.1.2 (2026-07-12)
------------------

Added
~~~~~

- Domain types: :class:`~mllm_shap.ConversationSnapshot`,
  :class:`~mllm_shap.Turn`, :class:`~mllm_shap.Message`,
  :class:`~mllm_shap.PlayerSet`, :class:`~mllm_shap.CoalitionMask`,
  :class:`~mllm_shap.CooperativeGame`, and conversation enums.
- Protocols for backends, estimators, value functions, partitions, absence
  policies, and embedding providers.
- :class:`~mllm_shap.PluginRegistry` with packaging entry-point discovery.
- :class:`~mllm_shap.ExplainConfig` (Pydantic v2) with YAML load/save helpers.
- Configuration guide and expanded API reference pages.

Documentation
~~~~~~~~~~~~~

- Sphinx theory pages for cooperative games and estimands (Shapley vs Banzhaf).
- Usage guide for estimand aggregators, results, and manifests.
- Expanded API reference for public modules.
- Example notebook ``examples/estimands_toy_game.ipynb``.
- Furo theme, getting-started page, embedded notebook rendering, and API module layout.

Tooling
~~~~~~~

- ``make notebooks`` target to execute example notebooks in place before commit.

.. _release-0-1-1:

0.1.1 (2026-07-12)
------------------

Added
~~~~~

- :class:`~mllm_shap.estimation.estimands.shapley.ShapleyAggregator` and
  :class:`~mllm_shap.estimation.estimands.banzhaf.BanzhafAggregator` for
  coalition-weighted attribution from masks and payoffs.
- :class:`~mllm_shap.domain.estimands.Estimand` enum and wire types for honest
  estimand labelling.
- :class:`~mllm_shap.pipeline.result.ExplainResult` with required
  ``estimand`` field.
- :class:`~mllm_shap.pipeline.manifest.RunManifest`,
  :class:`~mllm_shap.pipeline.manifest.RunManifestPayload`, and
  :func:`~mllm_shap.pipeline.manifest.parse_manifest` for run-archive metadata.
- :class:`~mllm_shap.protocols.estimand.EstimandAggregator` protocol.

.. _release-0-1-0:

0.1.0 (2026-07-12)
------------------

Added
~~~~~

- Initial PyPI package layout and ``mllm_shap`` import surface.
- Logging bootstrap from ``[tool.logging]`` in ``pyproject.toml`` via
  ``logging518``.
