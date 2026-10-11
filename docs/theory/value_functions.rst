Value functions and normalizers
===============================

Coalition **utility** is distinct from cooperative-game **value** (Shapley/Banzhaf
attributions). Value functions map model outputs to scalar payoffs; normalizers
reshape aggregated attributions for display only.

Utility families
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Plugin key
     - Utility
     - Mechanism
   * - ``tfidf_cosine``
     - Lexical similarity
     - Hashed token rows → frozen-corpus TF-IDF → cosine similarity
   * - ``embedding_cosine``
     - Semantic similarity
     - Cosine similarity of static or contextual embeddings
   * - ``embedding_euclidean``
     - Embedding distance
     - ``1 / (1 + L2 distance)`` between embeddings
   * - ``logprob``
     - Model confidence
     - :math:`v(S) = \frac{1}{|y_S|}\sum_t \log p(y_{S,t} \mid x_S, y_{S,<t})`,
       the mean log probability of the coalition's own generation :math:`y_S`
       given its prompt :math:`x_S`

:doc:`../guides/migrating_from_mllm_shap` maps these to the U1–U4 labels of
legacy MLLM-Shap.

Frozen TF-IDF
~~~~~~~~~~~~~

``tfidf_cosine`` freezes IDF weights on the full run corpus (the base generation
and every coalition output) before scoring. Every coalition comparison reuses
those weights, so payoffs within one run are on one scale.

Normalizers
-----------

Applied **after** :class:`~mllm_shap.estimation.estimands.shapley.ShapleyAggregator`
(or Banzhaf) aggregation:

* **Identity** — raw attributions
* **AbsSum** — divide by sum of absolute values
* **PowerShift** — shift to non-negative, power transform, scale to sum 1
* **MinMax** — scale to ``[0, 1]``, then scale to sum 1

Archives store generations (the base generation and every coalition output),
not payoffs or attributions. ``reanalyze`` recomputes payoffs with the runner's
value function and attributions with its normalizer, so either can be swapped
without new model calls.
