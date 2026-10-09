Examples
========

The ``examples/`` directory contains **Jupyter notebooks** with runnable
``nlp-shap`` workflows. They complement guides, theory pages, and the API
reference.

Setup
-----

Install the package:

.. code-block:: bash

   pip install nlp-shap

From a local clone:

.. code-block:: bash

   make install

``make notebooks`` re-executes every notebook and stores its outputs. For
interactive work, register the project environment as a kernel with
``make kernel`` (``nlp-shap (uv)``) or select ``.venv`` as the kernel.

Notebooks
---------

Each row links to the **rendered walkthrough** in these docs. Use **Source** to
open the ``.ipynb`` on GitHub or browse the full catalog in
`examples/README.md <https://github.com/Pawlo77/nlp-shap/blob/main/examples/README.md>`_.

.. list-table::
   :header-rows: 1
   :widths: 22 48 14 16

   * - Notebook
     - What it demonstrates
     - Extras
     - Source

   * - :doc:`notebooks/explain_pipeline`
     - :class:`~nlp_shap.ExplainRunner` end to end: content parts and
       snapshot JSON, :class:`~nlp_shap.PlayerSet` spans, YAML configuration
       and its :exc:`~nlp_shap.ConfigError` guards, the efficiency check,
       :meth:`~nlp_shap.ExplainRunOutput.to_dict`, every estimator and
       estimand, progress callbacks, stage spans and logging.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/explain_pipeline.ipynb>`__

   * - :doc:`notebooks/archive_reanalyze`
     - Run archives (manifest version 4, WAL, ``on_exists``),
       :meth:`~nlp_shap.ExplainRunner.reanalyze` reproducing explain,
       rescoring with every value function and normalizer, the archive
       errors and ``strict=True``.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/archive_reanalyze.ipynb>`__

   * - :doc:`notebooks/cli`
     - ``nlp-shap explain``, ``reanalyze``, ``plugins`` and ``config``; JSON
       and PNG output; exit codes 0, 2 and 3.
     - ``[viz]``
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/cli.ipynb>`__

   * - :doc:`notebooks/plugins_extending`
     - :func:`~nlp_shap.default_registry`, :class:`~nlp_shap.PluginContext`,
       ``plugin_options`` and their validation, a custom backend and value
       function, plugin conflicts and entry points.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/plugins_extending.ipynb>`__

   * - :doc:`notebooks/estimands_toy_game`
     - Shapley and Banzhaf aggregators on majority vs additive games;
       :class:`~nlp_shap.ExplainResult` labelling and
       :class:`~nlp_shap.RunManifest` wire format.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimands_toy_game.ipynb>`__

   * - :doc:`notebooks/masking_views`
     - Token partition, delete/pad/neutral absence policies,
       :class:`~nlp_shap.masking.MaskedSnapshot`,
       :class:`~nlp_shap.masking.MaskCodec`, and
       :class:`~nlp_shap.masking.MaskSpace`.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/masking_views.ipynb>`__

   * - :doc:`notebooks/runtime_core`
     - Run archive persistence, coalition dedup keys, hot LRU cache, and async
       :class:`~nlp_shap.runtime.InferenceScheduler` metrics.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/runtime_core.ipynb>`__

   * - :doc:`notebooks/exact_estimation`
     - :class:`~nlp_shap.ExactEstimator` enumeration of all :math:`2^n`
       coalitions and
       :meth:`~nlp_shap.ExactEstimator.estimate_attributions` estimand delegation
       with budget guards.
     - none
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/exact_estimation.ipynb>`__

   * - :doc:`notebooks/estimator_comparison`
     - Monte Carlo, complementary and Neyman-CC against exact Shapley values;
       relative L1 error as the budget fraction grows.
     - ``[viz]``
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/estimator_comparison.ipynb>`__

   * - :doc:`notebooks/attribution_viz`
     - Signed Shapley and Banzhaf values rendered with ``token_text`` and
       ``token_bar``, :func:`~nlp_shap.display_attribution_html`,
       :func:`~nlp_shap.render_attribution_html` and the renderer classes.
     - ``[viz]``
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/attribution_viz.ipynb>`__

   * - :doc:`notebooks/text_explain_e2e`
     - One refund prompt on the LM Studio, transformers (KV cache, ``dtype``,
       pinned ``revision``) and HTTP API backends from the ``config/``
       presets, then visualized with :func:`~nlp_shap.render_attribution`;
       LM Studio is skipped when no server runs.
     - ``[lmstudio]``, ``[transformers]``, ``[api]``, ``[viz]``
     - `GitHub <https://github.com/Pawlo77/nlp-shap/blob/main/examples/text_explain_e2e.ipynb>`__

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
