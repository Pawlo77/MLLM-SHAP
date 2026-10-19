Using value functions
=====================

A value function turns a coalition's model output into a payoff :math:`v(S)` by
comparing it with the base generation :math:`y_N` (the output for the full prompt).
Value functions are independent of the estimand (Shapley or Banzhaf) and the estimator
(how coalitions are sampled). ``explanation.value_fn`` selects one by name.

Choosing a value function
-------------------------

Pick by what the backend can return and what the answer looks like:

.. list-table::
   :header-rows: 1
   :widths: 8 44 20 28

   * - Use case
     - Situation
     - ``value_fn``
     - Measures
   * - U1
     - Logits available; fidelity to the explained model
     - ``tf_logprob``
     - Mean log probability of :math:`y_N` under the coalition prompt
   * - U2
     - Logits available; shifts among alternative answers
     - ``tf_jsd``
     - Jensen-Shannon shift of the next-token distributions over :math:`y_N`
   * - U3
     - Text output only, free-form answers
     - ``bertscore``
     - BERTScore F1 of the coalition answer against :math:`y_N`
   * - U4
     - Short factual or yes/no answers
     - ``nli_logodds``
     - Entailment log-odds in both directions; catches negation
   * - U5
     - One decision, such as refusing or choosing an option
     - ``contrastive``
     - Log-odds of a target continuation against a foil
   * - U6
     - Speech transcription
     - ``semascore``
     - Segment-aligned cosine times one minus character error
   * - U7
     - Spoken output
     - ``speech_bertscore``, ``transcribed``
     - SpeechBERTScore on audio; or any text value function on transcripts
   * - U8
     - Validation reference on subsets
     - ``judge_grade``
     - Expected grade of a judge model; never at dataset scale

The baselines stay available: ``tfidf_cosine`` (shared rare words, the
TokenSHAP-style comparison), ``embedding_cosine`` (sentence-embedding
similarity), ``reference_wer`` (word error rate against a reference) and
``self_logprob`` (confidence in the coalition's own answer). Every built-in declares
its contract, which the table below lists:

.. value-table::

The cost column is the work per coalition beyond the generation itself; the
teacher-forced functions and ``contrastive`` skip generating coalition answers
altogether. Value functions that a backend cannot serve fail with
:exc:`~mllm_shap.errors.ConfigError` before any generation. A cheaper function whose
attributions correlate above 0.9 with a costlier one can stand in for it; check that
on your data with :meth:`~mllm_shap.ExplainRunner.reanalyze` and
:meth:`~mllm_shap.ExplainRunner.rescore`.

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

In a run, ``encoder`` picks what is embedded. The default is the sentence-transformers
model ``all-MiniLM-L6-v2`` (extra ``mllm-shap[semantic]``): both answers are embedded
from their text, so ``explain`` and ``reanalyze`` agree and any backend works.
``encoder: model_states`` keeps the pooled states of the explained model, where
``explanation.embedding_mode`` (``static`` or ``contextual``) chooses the vector:

.. code-block:: yaml

   explanation:
     value_fn: embedding_cosine
     embedding_mode: contextual
     plugin_options:
       embedding_cosine: {encoder: model_states}

Pooled states of a generative model rank answer pairs poorly without similarity
training, which is why the sentence encoder is the default. Model states come from the
backend: the transformers backend returns the mean of the output tokens' input
embeddings (``static``) and of the last hidden state (``contextual``).
Backends without them (mock, LM Studio, HTTP API) would give 8-dimensional hash vectors
of the generated text, which carry no meaning, so the run fails with
:exc:`~mllm_shap.errors.ConfigError` unless
``plugin_options.embedding_cosine.allow_hash_embeddings`` is true (tests only).
:meth:`~mllm_shap.ExplainRunner.reanalyze` always scores hash vectors in
``model_states`` mode, because archives store the generated text only, so for a
transformers run its values differ from those of ``explain``; the manifest records
the encoder.

Settings of third-party value functions go under
``explanation.plugin_options.<name>``; see :doc:`extending`.

Self-confidence: ``self_logprob``
---------------------------------

.. code-block:: python

   from mllm_shap import GenerationRecord, SelfLogprobValue

   candidate = GenerationRecord(text="answer", logprobs=(-0.1, -0.2, -0.3))
   utility = SelfLogprobValue().score(candidate, candidate)
   print(round(utility, 4))

.. guide-result:: value_logprob

``v(S)`` is the mean natural-log probability of the tokens the model generated
for coalition ``S``, so it lies in ``(-inf, 0]`` and an empty generation scores 0.
It measures how confident the model is in its own answer and ignores the base
generation: a confident unrelated answer scores high. For fidelity to the full-input
answer use ``tf_logprob``. The name ``logprob`` is an alias for one minor version and
warns.

:class:`~mllm_shap.value.logprob.SelfLogprobValue` declares ``needs_logprobs``
(:class:`~mllm_shap.protocols.value.RequiresLogprobs`). A run with a backend that is
not :class:`~mllm_shap.protocols.backend.SupportsLogprobs` (``mock``, ``lmstudio``)
raises :exc:`~mllm_shap.errors.ConfigError` before any generation. A record that
still arrives without logprobs (an HTTP server that ignores the request) raises
:exc:`~mllm_shap.errors.LogprobsUnavailableError`. Archives store logprobs per
coalition, and :meth:`~mllm_shap.ExplainRunner.reanalyze` refuses ``self_logprob`` on
an archive whose rows have none. Backend support: :doc:`backends`.

Teacher-forced scores
---------------------

``tf_logprob``, ``tf_jsd`` and ``contrastive`` score a fixed continuation under each
coalition prompt with one forward pass and no generation of coalition answers:

- ``tf_logprob``: :math:`v(S) = \frac1\ell\sum_t \log p(y_{N,t} \mid y_{N,<t}, x_S)`,
  the target of Captum, ContextCite and TextGenSHAP. ``reduce: sum`` drops the factor
  :math:`1/\ell`.
- ``tf_jsd``: :math:`v(S) = -\sum_t \mathrm{JSD}(P_N \Vert P_S)` over the next-token
  distributions at each position of :math:`y_N`, so :math:`v(N) = 0`. ``top_k``
  (default 20) keeps that many candidates per position and buckets the rest.
- ``contrastive``: :math:`v(S) = \log p(t \mid x_S) - \log p(f \mid x_S)` for the
  ``target`` and ``foil`` options, such as ``" I can't"`` against ``" Sure"``.

.. code-block:: yaml

   explanation:
     value_fn: contrastive
     plugin_options:
       contrastive: {target: " I can't", foil: " Sure"}

The backend must implement :class:`~mllm_shap.protocols.backend.SupportsTeacherForcing`
(``transformers`` with text prompts, and ``mock``). The scores are archived per
coalition, so :meth:`~mllm_shap.ExplainRunner.reanalyze` reproduces ``explain``.
To score an archive written with another value function, such as ``tfidf_cosine``,
use :meth:`~mllm_shap.ExplainRunner.rescore`: it rebuilds the coalition prompts from
the archived masks and the conversation, runs forward passes only and issues no
``generate`` call.

.. code-block:: python

   import asyncio
   import tempfile
   from pathlib import Path

   from mllm_shap import (
       ConversationSnapshot, ExplainConfig, ExplainRunner, Message, Role, Turn,
   )

   snapshot = ConversationSnapshot.from_turns(
       (Turn((Message(role=Role.USER, text="alpha beta gamma"),)),)
   )

   def config(value_fn: str, archive: str = "") -> ExplainConfig:
       return ExplainConfig.model_validate({
           "backend": {"kind": "mock", "model_id": "mock"},
           "explanation": {
               "value_fn": value_fn,
               "absence_policy": "delete",
               "archive": {"path": archive},
           },
       })

   with tempfile.TemporaryDirectory() as tmp:
       text_run = ExplainRunner(config("tfidf_cosine", f"{tmp}/{{run_id}}")).explain_sync(
           snapshot
       )
       rescored = asyncio.run(
           ExplainRunner(config("tf_logprob")).rescore(snapshot, Path(tmp) / text_run.run_id)
       )
   print("backend generate calls:", rescored.metrics.executed)
   print("values:", [round(v, 3) for v in rescored.result.values])

.. guide-result:: value_rescore

Semantic scores from encoders
-----------------------------

- ``bertscore``: BERTScore F1 of greedy token matching against :math:`y_N`
  (``encoder``, ``layer``, optional ``baseline`` for rescaling). All answers of an
  instance are encoded in batches. Best text-only stand-in for log probability.
- ``nli_logodds``: with :math:`p` the geometric mean of the probabilities that each
  answer entails the other, :math:`v(S) = \log\frac{p}{1-p}`. A negated answer scores
  low where similarity stays high. ``question`` is prepended to both answers.
- ``semascore``: transcripts aligned into segments; each segment scores the cosine of
  its contextual word vectors times one minus its character error, weighted by its
  closeness to the whole reference. ``reference`` works as in ``reference_wer``.

Their encoders load lazily (extra ``mllm-shap[semantic]``) and are recorded in the
manifest. Unit tests and the snippet below inject small encoders instead:

.. code-block:: python

   import numpy as np
   from mllm_shap import BertScoreValue, GenerationRecord

   class Axes:
       vectors = {"cat": (1.0, 0.0), "dog": (0.0, 1.0), "pet": (0.6, 0.8)}

       def encode(self, texts):
           return [np.array([self.vectors[w] for w in t.split()]) for t in texts]

   value_fn = BertScoreValue(encoder=Axes())
   score = value_fn.score(GenerationRecord(text="cat dog"), GenerationRecord(text="pet"))
   print(round(score, 4))

.. guide-result:: value_bertscore

Spoken answers
--------------

``speech_bertscore`` compares the audio of the coalition answer with the full-input
answer: precision of greedy cosine matching over WavLM frames, so recordings need not
align (extra ``mllm-shap[audio-semantic]``). ``transcribed`` wraps a text value
function: a fixed recognizer transcribes both answers and the wrapped function scores
the transcripts, which removes how the answer sounds:

.. code-block:: yaml

   explanation:
     value_fn: transcribed
     plugin_options:
       transcribed: {value_fn: bertscore, recognizer: openai/whisper-base}

Both read ``GenerationRecord.audio_path``, which a backend that returns speech sets;
no built-in backend does yet, so these are used with a custom backend.

Judge reference
---------------

``judge_grade`` asks a judge model how well a coalition answer preserves the meaning
of :math:`y_N` on a 1 to 5 scale and returns the probability-weighted grade, which is
continuous and so has few ties. The judge is any registered backend named by
``backend`` and ``model_id`` options. With teacher forcing it reads the probabilities
of the digits; otherwise ``samples`` graded generations are averaged. It costs a model
call per distinct answer and refuses runs above ``max_coalitions`` (2000) answers
unless ``allow_large_judge`` is set: use it to check the cheap functions on a subset.

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
