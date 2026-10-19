Value functions and normalizers
===============================

Coalition **utility** is distinct from cooperative-game **value** (Shapley/Banzhaf
attributions). Value functions map model outputs to scalar payoffs; normalizers
reshape aggregated attributions for display only.

Contract and families
---------------------

Every built-in declares the range of its score, whether it reads the full-input answer
:math:`y_N`, what it requires of the backend, any extra model and its cost per
coalition (:class:`~mllm_shap.protocols.value.ValueInfo`). The table is generated
from those declarations:

.. value-table::

:doc:`../guides/value_functions` maps use cases to functions. The formulas, with
:math:`x_S` the prompt with players :math:`S` present, :math:`y_S` its generated
answer and :math:`y_N` the answer for the full prompt:

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Plugin key
     - Utility
     - Reference
   * - ``tfidf_cosine``
     - Cosine of TF-IDF vectors of :math:`y_S` and :math:`y_N`, IDF frozen on the run
     - TokenSHAP
   * - ``embedding_cosine``
     - Cosine of sentence embeddings of :math:`y_S` and :math:`y_N` (default
       ``all-MiniLM-L6-v2``), or of the model's pooled states
     - Reimers and Gurevych 2019 (`arXiv:1908.10084 <https://arxiv.org/abs/1908.10084>`_)
   * - ``self_logprob``
     - :math:`\frac{1}{|y_S|}\sum_t \log p(y_{S,t} \mid x_S, y_{S,<t})`, the
       confidence in the coalition's own answer
     -
   * - ``tf_logprob``
     - :math:`\frac{1}{\ell}\sum_t \log p(y_{N,t} \mid y_{N,<t}, x_S)`, the
       teacher-forced log probability of :math:`y_N` (``sum`` omits :math:`1/\ell`)
     - MExGen (`arXiv:2403.14459 <https://arxiv.org/abs/2403.14459>`_), Captum
       (`arXiv:2312.05491 <https://arxiv.org/abs/2312.05491>`_)
   * - ``tf_jsd``
     - :math:`-\sum_t \mathrm{JSD}\big(P(\cdot \mid y_{N,<t}, x_N)\,\Vert\,
       P(\cdot \mid y_{N,<t}, x_S)\big)`
     - ARC-JSD (`arXiv:2505.16415 <https://arxiv.org/abs/2505.16415>`_)
   * - ``contrastive``
     - :math:`\log p(t \mid x_S) - \log p(f \mid x_S)` for a target :math:`t` and
       a foil :math:`f`
     - Yin and Neubig 2022 (`arXiv:2202.10419 <https://arxiv.org/abs/2202.10419>`_)
   * - ``bertscore``
     - :math:`F_1` of greedy token matching of encoder states, :math:`R =
       \frac{1}{|y_N|}\sum_i \max_j \cos(e_i, \hat e_j)`
     - Zhang et al. 2020 (`arXiv:1904.09675 <https://arxiv.org/abs/1904.09675>`_)
   * - ``nli_logodds``
     - :math:`\log\frac{p}{1-p}` with :math:`p = \sqrt{p_{N\to S}\,p_{S\to N}}`,
       the entailment probabilities
     - Kuhn et al. 2023 (`arXiv:2302.09664 <https://arxiv.org/abs/2302.09664>`_),
       MExGen
   * - ``semascore``
     - :math:`\sum_i \alpha_i \cos(e_{g_i}, e_{h_i})(1-\mathrm{MER}_i) /
       \sum_i \alpha_i`, :math:`\alpha_i = \cos(e_{g_i}, e_y)`
     - Sasindran et al. 2024 (`arXiv:2401.07506 <https://arxiv.org/abs/2401.07506>`_)
   * - ``speech_bertscore``
     - :math:`\frac{1}{N_S}\sum_i \max_j \cos(\hat z_i, z_j)` over WavLM frames
     - Saeki et al. 2024 (`arXiv:2401.16812 <https://arxiv.org/abs/2401.16812>`_)
   * - ``transcribed``
     - A text value function on recogniser transcripts of :math:`y_S`, :math:`y_N`
     -
   * - ``judge_grade``
     - :math:`\sum_{g=1}^{5} g\,p(g \mid \text{judge prompt})`
     - G-Eval (`arXiv:2303.16634 <https://arxiv.org/abs/2303.16634>`_)
   * - ``reference_wer``
     - :math:`1 - \min(1, \mathrm{WER}(y_S, \text{reference}))`
     -

``tf_jsd`` bounds each position by :math:`\log 2`. It keeps the ``top_k`` most
probable tokens of each distribution plus one bucket for the remaining mass, so it is
exact when ``top_k`` covers the vocabulary and a lower bound otherwise.

Fixed payoffs
~~~~~~~~~~~~~

At temperature above zero two generations of one coalition differ, and Shapley values
lose efficiency. The pipeline generates each distinct coalition once per run and
reuses it (``dedup.enabled: auto`` is on at any temperature), so
:math:`\sum_i \phi_i = v(N) - v(\emptyset)` holds for exact estimators; the manifest
records temperature, top-k, seed and the deduplication policy.

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

Archives store generations (the base generation and every coalition output)
and any teacher-forced scores, not payoffs or attributions. ``reanalyze`` recomputes
payoffs with the runner's value function and attributions with its normalizer, so
either can be swapped without new model calls; ``rescore`` adds forward passes for the
teacher-forced functions and still never generates.
