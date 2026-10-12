Examples
========

The ``examples/`` directory contains **Jupyter notebooks** with runnable
``mllm-shap`` workflows. They complement guides, theory pages, and the API
reference.

Setup
-----

Install the package:

.. code-block:: bash

   pip install mllm-shap

From a local clone:

.. code-block:: bash

   make install

``make notebooks`` re-executes every notebook and stores its outputs. For
interactive work, register the project environment as a kernel with
``make kernel`` (``mllm-shap (uv)``) or select ``.venv`` as the kernel.

Notebooks
---------

Each row links to the **rendered walkthrough** in these docs. Use **Source** to
open the ``.ipynb`` on GitHub or browse the full catalog in
`examples/README.md <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/README.md>`_.

.. list-table::
   :header-rows: 1
   :widths: 22 48 14 16

   * - Notebook
     - What it demonstrates
     - Extras
     - Source

   * - :doc:`notebooks/explain_pipeline`
     - :class:`~mllm_shap.ExplainRunner` end to end: content parts and
       snapshot JSON, :class:`~mllm_shap.PlayerSet` spans, YAML configuration
       and its :exc:`~mllm_shap.ConfigError` guards, the efficiency check,
       :meth:`~mllm_shap.ExplainRunOutput.to_dict`, every estimator and
       estimand, progress callbacks, stage spans and logging.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/explain_pipeline.ipynb>`__

   * - :doc:`notebooks/archive_reanalyze`
     - Run archives (manifest version 4, WAL, ``on_exists``),
       :meth:`~mllm_shap.ExplainRunner.reanalyze` reproducing explain,
       rescoring with every value function and normalizer, the archive
       errors and ``strict=True``.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/archive_reanalyze.ipynb>`__

   * - :doc:`notebooks/cli`
     - ``mllm-shap explain``, ``reanalyze``, ``plugins`` and ``config``; JSON
       and PNG output; exit codes 0, 2 and 3.
     - ``[viz]``
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/cli.ipynb>`__

   * - :doc:`notebooks/plugins_extending`
     - :func:`~mllm_shap.default_registry`, :class:`~mllm_shap.PluginContext`,
       ``plugin_options`` and their validation, a custom backend and value
       function, plugin conflicts and entry points.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/plugins_extending.ipynb>`__

   * - :doc:`notebooks/estimands_toy_game`
     - Shapley and Banzhaf aggregators on majority vs additive games;
       :class:`~mllm_shap.ExplainResult` labelling and
       :class:`~mllm_shap.RunManifest` wire format.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/estimands_toy_game.ipynb>`__

   * - :doc:`notebooks/masking_views`
     - Token partition, delete/pad/neutral absence policies,
       :class:`~mllm_shap.masking.MaskedSnapshot`,
       :class:`~mllm_shap.masking.MaskCodec`, and
       :class:`~mllm_shap.masking.MaskSpace`.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/masking_views.ipynb>`__

   * - :doc:`notebooks/runtime_core`
     - Run archive persistence, coalition dedup keys, hot LRU cache, and async
       :class:`~mllm_shap.runtime.InferenceScheduler` metrics.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/runtime_core.ipynb>`__

   * - :doc:`notebooks/exact_estimation`
     - :class:`~mllm_shap.ExactEstimator` enumeration of all :math:`2^n`
       coalitions and
       :meth:`~mllm_shap.ExactEstimator.estimate_attributions` estimand delegation
       with budget guards.
     - none
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/exact_estimation.ipynb>`__

   * - :doc:`notebooks/estimator_comparison`
     - Monte Carlo, complementary and Neyman-CC against exact Shapley values;
       relative L1 error as the budget fraction grows.
     - ``[viz]``
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/estimator_comparison.ipynb>`__

   * - :doc:`notebooks/attribution_viz`
     - Real-model attributions of a prompt (words), a multi-speaker
       conversation (messages, Shapley and Banzhaf) and a spoken sentence (audio
       words), rendered with ``token_text``, ``token_bar``,
       :func:`~mllm_shap.display_attribution_html`,
       :func:`~mllm_shap.render_attribution_html`, the renderer classes and
       :func:`~mllm_shap.viz.plot_audio_segments`.
     - ``[viz]``, ``[transformers]``, ``[audio]``
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/attribution_viz.ipynb>`__

   * - :doc:`notebooks/text_explain_e2e`
     - One refund prompt on the LM Studio, transformers (KV cache, ``dtype``,
       pinned ``revision``) and HTTP API backends from the ``config/``
       presets, then visualized with :func:`~mllm_shap.render_attribution`;
       LM Studio is skipped when no server runs.
     - ``[lmstudio]``, ``[transformers]``, ``[api]``, ``[viz]``
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/text_explain_e2e.ipynb>`__

   * - :doc:`notebooks/audio_alignment`
     - :class:`~mllm_shap.alignment.SpectrogramGuidedAligner` on a recorded
       sentence with a pinned Wav2Vec2 model: word segments, shared refined
       boundaries, boundary weights, attached segment audio, the
       ``ValueError`` guards and a custom :class:`~mllm_shap.alignment.CtcModel`.
     - ``[audio]``, ``[transformers]``
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/audio_alignment.ipynb>`__

   * - :doc:`notebooks/mock_audio_backend`
     - :class:`~mllm_shap.backends.mock.audio.MockAudioBackend` on a synthesised clip: heard tokens
       per voiced run, a silenced word, ``max_new_tokens``, an explanation with
       ``backend.kind: mock_audio`` and the undecodable-audio guard.
     - *(core)*
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/mock_audio_backend.ipynb>`__

   * - :doc:`notebooks/audio_explain_transformers`
     - Granite Speech hears a recorded sentence through the ``transformers``
       backend, then ``sgpa_segments`` attributes its answer to the spoken words;
       the ``lmstudio`` audio refusal.
     - ``[audio]``, ``[transformers]``
     - `GitHub <https://github.com/Pawlo77/MLLM-SHAP/blob/main/examples/audio_explain_transformers.ipynb>`__

.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/explain_pipeline
   notebooks/archive_reanalyze
   notebooks/cli
   notebooks/plugins_extending
   notebooks/estimands_toy_game
   notebooks/masking_views
   notebooks/runtime_core
   notebooks/exact_estimation
   notebooks/estimator_comparison
   notebooks/attribution_viz
   notebooks/text_explain_e2e
   notebooks/audio_alignment
   notebooks/mock_audio_backend
   notebooks/audio_explain_transformers
