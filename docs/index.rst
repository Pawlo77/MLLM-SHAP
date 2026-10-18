mllm-shap
=========

Shapley-value attributions for conversational language models. ``mllm-shap``
separates *what you measure* (the estimand: Shapley or Banzhaf), *how you sample
coalitions* (the estimator) and *how you score outputs* (the value function), and
archives every run so it can be reproduced and re-analysed.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: guides/getting_started
      :link-type: doc

      Install ``mllm-shap`` and explain a first prompt, then check that the values
      add up.

   .. grid-item-card:: How a run works
      :link: guides/pipeline
      :link-type: doc

      The stages of an explanation run, its counters, archives and hooks.

   .. grid-item-card:: Theory
      :link: theory/shapley_values
      :link-type: doc

      Shapley axioms, cooperative games, estimands, and when they diverge.

   .. grid-item-card:: Applications
      :link: guides/applications
      :link-type: doc

      Business, compliance, and LLM operations use cases for attribution.

   .. grid-item-card:: API reference
      :link: api
      :link-type: doc

      Typed public modules, protocols, and re-exported symbols.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: doc

      Runnable Jupyter notebooks with end-to-end walkthroughs (estimands,
      masking, runtime, exact and approximate estimation).

   .. grid-item-card:: Citing
      :link: citing
      :link-type: doc

      How to cite ``mllm-shap`` and the methods behind it.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      Development setup, quality gates and pull request conventions.

   .. grid-item-card:: Knowledge vault
      :link: https://github.com/Pawlo77/MLLM-SHAP/blob/main/knowledge/mllm-shap.md

      Architecture, design decisions, known issues and papers, kept as an
      Obsidian vault in the repository.

How it compares
---------------

``shap`` (``KernelExplainer``, ``PartitionExplainer``), ``captum``
(``ShapleyValueSampling``) and TokenSHAP also estimate Shapley values for text.
``mllm-shap`` differs in four ways:

- the estimand (Shapley or Banzhaf) is an explicit label on every result, separate
  from the estimator (:doc:`theory/estimands`);
- every run is archived and can be rescored without backend calls
  (:meth:`~mllm_shap.ExplainRunner.reanalyze`, :doc:`guides/runtime`);
- backends are plugins: mock, LM Studio, Hugging Face ``transformers`` and
  OpenAI-compatible APIs (:doc:`guides/backends`);
- complementary-contribution and Neyman estimators sit next to Monte Carlo and
  exact enumeration (:doc:`theory/approximation`).

Quick install
-------------

.. code-block:: bash

   pip install mllm-shap

Requires Python 3.12 or newer. See :doc:`guides/getting_started` for a full
walkthrough.

Optional extras:

.. code-block:: bash

   pip install "mllm-shap[transformers]"   # Hugging Face backend
   pip install "mllm-shap[lmstudio]"       # LM Studio backend
   pip install "mllm-shap[api]"            # OpenAI-compatible HTTP API
   pip install "mllm-shap[viz]"            # matplotlib attribution charts

See :doc:`guides/extending` for the full extras and entry-point matrix.

.. toctree::
   :hidden:
   :caption: Guides
   :maxdepth: 2

   guides/getting_started
   guides/pipeline
   guides/cli
   guides/config
   guides/estimands
   guides/exact
   guides/approximation
   guides/choosing_an_estimator
   guides/estimator_benchmark
   guides/value_functions
   guides/masking
   guides/backends
   guides/audio_backends
   guides/runtime
   guides/visualization
   guides/alignment
   guides/extending
   guides/applications
   guides/migrating_from_mllm_shap

.. toctree::
   :hidden:
   :caption: Theory
   :maxdepth: 2

   theory/cooperative_games
   theory/shapley_values
   theory/estimands
   theory/exact
   theory/approximation
   theory/regression_estimators
   theory/value_functions
   theory/masking
   theory/runtime

.. toctree::
   :hidden:
   :caption: Reference
   :maxdepth: 2

   api
   examples
   release_notes
   citing
   contributing
   handbook/index
