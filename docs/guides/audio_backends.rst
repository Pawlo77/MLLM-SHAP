Audio input
===========

A message can carry :class:`~mllm_shap.AudioPart` content next to its text. With
``players: sgpa_segments`` (:doc:`alignment`) the absence policies cut or silence
the samples of absent words, and the backend decides what the model hears.

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - ``kind``
     - Extra
     - Audio parts
   * - ``transformers``
     - ``[transformers,audio]``
     - Sent to audio language models through their processor
   * - ``api``
     - ``[api]``
     - Sent as OpenAI ``input_audio`` parts (base64 WAV)
   * - ``lmstudio``
     - ``[lmstudio]``
     - Refused by default; ``audio: transcript`` sends the text parts only
   * - ``mock_audio``
     - *(core)*
     - One token per voiced run, for tests and demos

Audio language models in transformers
-------------------------------------

:class:`~mllm_shap.backends.transformers.TransformersTextBackend` reads the
checkpoint's config. A model ``AutoModelForCausalLM`` knows loads as a text model.
Any other loads its own class (``architectures``, or
``backend.options.model_class``) with ``AutoProcessor`` and takes audio parts; a
text model refuses them with ``ValueError``. Set ``model_class`` for a multimodal
model that ``AutoModelForCausalLM`` also knows, such as Gemma 3n.

- Each audio part is decoded and resampled to the processor's sampling rate.
- A processor whose chat template mentions audio gets structured parts: the audio
  of a message first, then its text. Otherwise each audio part becomes the
  processor's ``audio_token`` in the rendered prompt and the waveforms go to
  ``processor(text=..., audio=...)``; Granite Speech works this way.
- An audio part that masking left empty is dropped with its token.
- Audio prompts skip the prefix cache: cached prompt tokens do not cover the
  audio features. Logprobs and embeddings work as for text.

.. code-block:: yaml

   backend:
     kind: transformers
     model_id: ibm-granite/granite-speech-3.3-2b
     options:
       revision: 4ac2f02f413c6169ae8c0ccc217115a366e552d7
       dtype: bfloat16
       attn_implementation: eager  # Granite's projector fails under SDPA on MPS

   explanation:
     players: sgpa_segments
     absence_policy: delete

Granite needs ``torchaudio`` (features) and ``peft`` (its audio adapter), both in the ``transformers``
extra. ``backend.options.attn_implementation`` is passed to ``from_pretrained``. Notebook: :doc:`/notebooks/audio_explain_transformers`.

HTTP API
--------

:class:`~mllm_shap.backends.api.ApiBackend` sends a message with audio as a list
of parts: its text first, then one ``input_audio`` part per audio part. WAV bytes go
as they are; other formats are re-encoded to WAV (``[audio]`` extra).

.. code-block:: json

   {"role": "user", "content": [
     {"type": "text", "text": "Transcribe this recording."},
     {"type": "input_audio", "input_audio": {"data": "UklGR...", "format": "wav"}}
   ]}

A server that takes no audio answers with an error status, which raises
:exc:`~mllm_shap.errors.BackendUnavailableError`.

LM Studio
---------

The LM Studio SDK takes images but no audio, so
:class:`~mllm_shap.backends.lmstudio.LmStudioBackend` refuses audio parts with
``ValueError``. ``backend.options.audio: transcript`` sends the text parts only
and logs one warning: the run then explains the masked transcript, not the sound.

Mock
----

:class:`~mllm_shap.backends.mock.audio.MockAudioBackend` (``mock_audio``) echoes the
words of each message, then adds one token per voiced run of each audio part: a
stretch louder than 1% of full scale, split by at least 10 ms of silence. A token
hashes its run's samples, so silencing or cutting a spoken word drops only its
token. It reads 16-bit WAV without any extra; other formats need ``[audio]``.
Notebook: :doc:`/notebooks/mock_audio_backend`.
