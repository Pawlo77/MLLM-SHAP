Coming from legacy MLLM-Shap
============================

``mllm-shap`` replaces the `legacy MLLM-Shap <https://github.com/Pawlo77/MLLM-Shap-legacy>`_
research code. The algorithms are the same family; the configuration, the result
types and a few defaults differ. This page maps the old names to the new ones.

Explainers to estimators
------------------------

legacy MLLM-Shap had one explainer class per estimator and sampling variant. In
``mllm-shap`` the estimator is a name in ``explanation.estimator`` and the variant
is expressed with the budget keys:

.. list-table::
   :header-rows: 1
   :widths: 42 28 30

   * - legacy MLLM-Shap class
     - ``explanation.estimator``
     - Budget keys
   * - ``PreciseShapExplainer``
     - ``exact``
     - ``budget.fraction: 1.0``
   * - ``StandardMcShapExplainer``
     - ``mc``
     - ``fraction: 1.0``, ``include_minimal_masks: false``
   * - ``LimitedMcShapExplainer``
     - ``mc``
     - ``fraction < 1.0`` and/or ``include_minimal_masks: true``
   * - ``StandardComplementaryShapExplainer``
     - ``complementary``
     - ``fraction: 1.0``, ``include_minimal_masks: false``
   * - ``LimitedComplementaryShapExplainer``
     - ``complementary``
     - ``fraction < 1.0`` and/or ``include_minimal_masks: true``
   * - ``StandardComplementaryNeymanShapExplainer``
     - ``neyman_cc``
     - ``fraction: 1.0``
   * - ``LimitedComplementaryNeymanShapExplainer``
     - ``neyman_cc``
     - ``fraction < 1.0``

The estimand is a separate key: ``explanation.estimand`` is ``shapley`` or
``banzhaf`` and every result carries the label (:doc:`estimands`).

Utilities to value functions
----------------------------

The legacy MLLM-Shap experiments named their utilities U1 to U4. The corresponding
``explanation.value_fn`` settings:

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - Label
     - legacy MLLM-Shap
     - ``mllm-shap``
   * - U1
     - ``CosineSimilarity`` on static embeddings
     - ``value_fn: embedding_cosine``, ``plugin_options.embedding_cosine.encoder:
       model_states``, ``embedding_mode: static``
   * - U2
     - ``CosineSimilarity`` on contextual embeddings
     - ``value_fn: embedding_cosine``, ``plugin_options.embedding_cosine.encoder:
       model_states``, ``embedding_mode: contextual``
   * - U3
     - ``TfIdfCosineSimilarity``
     - ``value_fn: tfidf_cosine``
   * - U4
     - ``EuclideanSimilarity``
     - None: ``embedding_euclidean`` was removed in 0.7 because it depends on vector
       scale and answer length; use ``embedding_cosine``

Since 0.7 ``embedding_cosine`` embeds the answers with a sentence encoder by
default; ``model_states`` keeps the legacy pooled states of the explained model.

Behaviour that changed
----------------------

- **Frozen TF-IDF.** ``tfidf_cosine`` fits its IDF weights once on all outputs of
  the run and scores every coalition with them, instead of refitting per
  comparison. Scores are therefore comparable across coalitions of one run.
- **Efficiency is verifiable.** Results report ``base_value`` and ``grand_value``
  (the empty and grand coalitions), so you can check that the values sum to their
  difference; the axiom tests enforce it. Nothing checks it at run time. Exact results match ``PreciseShapExplainer`` on additive
  games.
- **Archives store generations.** A run archive keeps the base generation and
  every coalition output; :meth:`~mllm_shap.ExplainRunner.reanalyze` recomputes
  payoffs and attributions with another value function or normalizer without new
  model calls.
- **Mask hashes are process-independent.**
  :meth:`~mllm_shap.masking.codec.MaskCodec.hash_mask` uses BLAKE2b, so hashes
  written by legacy MLLM-Shap are not comparable with the ones ``mllm-shap`` produces.
- **One configuration object.** Model, generation and explanation settings live in
  one validated :class:`~mllm_shap.ExplainConfig` (:doc:`config`) instead of
  separate keyword arguments and model config objects.
