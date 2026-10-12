Using value functions
=====================

A value function turns a coalition's model output into a payoff :math:`v(S)` by
comparing it with the base generation (the output for the full prompt). Value
functions are independent of the estimand (Shapley or Banzhaf) and the estimator
(how coalitions are sampled). ``explanation.value_fn`` selects one by name.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - ``value_fn``
     - Payoff of a coalition output
   * - ``tfidf_cosine``
     - Cosine similarity of TF-IDF vectors of the output and the base generation,
       with IDF weights frozen on all outputs of the run
   * - ``embedding_cosine``
     - Cosine similarity of the output and base embeddings (static or contextual)
   * - ``embedding_euclidean``
     - ``1 / (1 + distance)`` between the output and base embeddings
   * - ``logprob``
     - Mean log probability of the output's own tokens; needs a backend that
       returns them
   * - ``reference_wer``
     - ``1 - min(1, WER)`` of the output against ``plugin_options.reference``
       (such as a transcript), or against the base generation without one

:doc:`migrating_from_mllm_shap` maps these to the U1–U4 utility labels used by
legacy MLLM-Shap.

TF-IDF cosine
-------------

.. code-block:: python

   from mllm_shap import GenerationRecord, TfIdfCosineValue

   base = GenerationRecord(
       text="hello world",
       text_token_rows=((1, 0), (0, 1)),
   )
   candidate = GenerationRecord(
       text="hello there",
       text_token_rows=((1, 0), (2, 3)),
   )

   value_fn = TfIdfCosineValue()
   value_fn.fit((base, candidate))
   utility = value_fn.score(base, candidate)
   print(round(utility, 4))

.. guide-result:: value_tfidf

The pipeline calls :meth:`~mllm_shap.value.tfidf.TfIdfCosineValue.fit` once on
every output of the run, so all coalitions are scored with the same IDF weights,
and :meth:`~mllm_shap.ExplainRunner.reanalyze` refits on the archived outputs.

Embedding similarity
--------------------

.. code-block:: python

   from mllm_shap import CosineEmbeddingValue, GenerationRecord
   from mllm_shap.domain.enums import EmbeddingMode

   base = GenerationRecord(text="base", embedding=(1.0, 0.0))
   candidate = GenerationRecord(text="candidate", embedding=(0.9, 0.1))

   value_fn = CosineEmbeddingValue(embedding_mode=EmbeddingMode.STATIC)
   similarity = value_fn.score(base, candidate)
   print(round(similarity, 4))

.. guide-result:: value_embedding

In a run the mode comes from ``explanation.embedding_mode`` (``static`` or
``contextual``): the registry builds the value function with a
:class:`~mllm_shap.plugins.context.PluginContext` and the plugin reads the setting
from it, so the YAML below is all you need:

.. code-block:: yaml

   explanation:
     value_fn: embedding_cosine
     embedding_mode: contextual

Where the vectors come from depends on the backend. The embedding value functions
declare ``needs_embeddings`` (:class:`~mllm_shap.protocols.value.RequiresEmbeddings`),
so the pipeline asks the backend for model embeddings
(``GenerationConfig.compute_embeddings``): the transformers backend returns the
mean of the output tokens' input embeddings (``static``) and of the last hidden
state (``contextual``), and ``explain`` scores those. Backends without embeddings
(mock, LM Studio, HTTP API) give deterministic hash vectors of the generated text,
which only make sense for tests. :meth:`~mllm_shap.ExplainRunner.reanalyze` always
scores the hash vectors, because archives store the generated text only, so for a
transformers run its embedding values differ from those of ``explain``.

Settings of third-party value functions go under
``explanation.plugin_options.<name>``; see :doc:`extending`.

Logprob utility
---------------

.. code-block:: python

   from mllm_shap import GenerationRecord, LogprobValue

   candidate = GenerationRecord(text="answer", logprobs=(-0.1, -0.2, -0.3))
   utility = LogprobValue().score(candidate, candidate)
   print(round(utility, 4))

.. guide-result:: value_logprob

``v(S)`` is the mean natural-log probability of the tokens the model generated
for coalition ``S``, so it lies in ``(-inf, 0]`` and an empty generation scores 0.
It measures how confident the model is in its own answer; the base generation is
not read, and ``v(N)`` is the base generation's own mean.

:class:`~mllm_shap.value.logprob.LogprobValue` declares ``needs_logprobs``
(:class:`~mllm_shap.protocols.value.RequiresLogprobs`). A run with a backend that is
not :class:`~mllm_shap.protocols.backend.SupportsLogprobs` (``mock``, ``lmstudio``)
raises :exc:`~mllm_shap.errors.ConfigError` before any generation. A record that
still arrives without logprobs (an HTTP server that ignores the request) raises
:exc:`~mllm_shap.errors.LogprobsUnavailableError`. Archives store logprobs per
coalition, and :meth:`~mllm_shap.ExplainRunner.reanalyze` refuses ``logprob`` on
an archive whose rows have none. Backend support: :doc:`backends`.

Word error rate and deletion curves
-----------------------------------

``reference_wer`` scores how correct an answer is: with
``plugin_options: {reference_wer: {reference: "i had that curiosity"}}`` a
transcription prompt earns 1 for the exact transcript, ignoring case and
punctuation. :func:`~mllm_shap.value.wer.word_error_rate` is the metric alone.

:func:`~mllm_shap.deletion_curve` tests an attribution's faithfulness. It removes
the ``k`` highest-ranked players at once, for each ``k``, and rescores the
coalition left. A lower :attr:`~mllm_shap.DeletionCurve.area` means the value
falls sooner. Score the curve with a different value function than the
explanation, or the check is circular. Pass the explained run's ``backend`` to
reuse its model:

.. code-block:: python

   curve = await deletion_curve(wer_config, snapshot, output.result.values)
   print(curve.removed, curve.values, round(curve.area, 3))

Attribution normalizers
-----------------------

Normalizers rescale aggregated attributions for presentation. They do **not**
change coalition payoffs, so ``base_value`` and ``grand_value`` stay comparable
across normalizers, and efficiency only holds for ``identity``.

.. code-block:: python

   from mllm_shap import MinMaxNormalizer

   raw = [1.0, -2.0, 3.0]
   displayed = MinMaxNormalizer().normalize(raw)
   print(displayed)

.. guide-result:: value_normalizer

Config keys: ``identity``, ``abs_sum``, ``power_shift``, ``min_max``. See
:doc:`config` and :doc:`../theory/value_functions`.
``examples/archive_reanalyze.ipynb`` (embedded on :doc:`../examples`) rescores
one archived run with every value function and normalizer.
