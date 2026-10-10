Audio alignment
===============

:class:`~nlp_shap.alignment.SpectrogramGuidedAligner` turns a transcript and the
speech that says it into one :class:`~nlp_shap.alignment.AudioSegment` per token.
Segments are the unit an audio explanation masks, so a cut must not split a sound:
the aligner places each boundary at the quietest, steadiest point near the gap
between two words.

Install
-------

.. code-block:: bash

   pip install "nlp-shap[audio,transformers]"

The ``audio`` extra decodes audio (``soundfile``) and refines boundaries
(``librosa``); the ``transformers`` extra runs the default Wav2Vec2 model.

Align a transcript
------------------

.. code-block:: python

   from pathlib import Path

   from nlp_shap import AudioPart
   from nlp_shap.alignment import SpectrogramGuidedAligner, Wav2Vec2Ctc

   model = Wav2Vec2Ctc("facebook/wav2vec2-base-960h", revision="main", device="cpu")
   aligner = SpectrogramGuidedAligner(model)
   audio = AudioPart(Path("speech.wav"), sample_rate=16000)
   for segment in aligner.align("I had that curiosity beside me", audio):
       print(segment.token, round(segment.start_time, 2), round(segment.end_time, 2))

``align`` accepts a string (split on whitespace) or a list of tokens. A token with
no character the model can spell, such as a dash, gets no segment.
``attach_audio=True`` stores each segment's samples as WAV bytes in
``segment.audio``.

How it works
------------

1. The CTC (connectionist temporal classification) model scores every audio frame.
2. A Viterbi pass aligns the transcript's characters to frames in order,
   with the model's word separator between tokens.
3. A token spans its first to last character. Each boundary then moves, within
   the gap between two tokens, to the minimum of
   ``energy_weight * rms + flux_weight * flux``: the normalised RMS (root mean
   square) energy and spectral flux of the waveform. The first and last edges
   search 80 ms around their rough position.
4. A boundary moves only when that point is quiet (RMS at most half the window's
   mean); ``AudioSegment.boundary_refined`` says whether both edges moved.

Segments last at least 50 ms. Times are in seconds and sample indices refer to the
audio's own sample rate.

Audio players
-------------

``explanation.players: sgpa_segments``
(:class:`~nlp_shap.SgpaSegmentPartitioner`) makes each spoken word a player. A
message qualifies when it holds an :class:`~nlp_shap.AudioPart` and its transcript
as text; other messages add no players. Each player's
:class:`~nlp_shap.PlayerSpan` covers the word in the transcript and, in
``samples``, its stretch of audio, so the absence policies mask both: ``delete``
cuts the absent samples, ``pad`` and ``neutral`` replace them with silence of the
same length.

.. code-block:: python

   from nlp_shap import ExplainConfig, Message, Role, TextPart

   spoken = Message(role=Role.USER, content=(TextPart("I had that curiosity"), audio))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "players": "sgpa_segments",
           "plugin_options": {"sgpa_segments": {
               "model_id": "facebook/wav2vec2-base-960h", "revision": "main"}},
       },
   })

Plugin options: ``model_id``, ``revision``, ``device``, ``energy_weight`` and
``flux_weight``. No backend generates from audio yet, so a run scores the masked
transcripts.

Other CTC models
----------------

Any object with ``sample_rate``, ``blank_id``, ``vocab`` and
``log_probs(waveform)`` satisfies :class:`~nlp_shap.alignment.CtcModel`, for
example another Wav2Vec2 checkpoint for a different language, or a fixed matrix
in tests. Pin ``revision`` to a commit hash for reproducible segments.

The :doc:`../notebooks/audio_alignment` notebook runs every step on a recorded
sentence. The method follows SGPA (spectrogram-guided phonetic alignment),
`arXiv:2603.02250 <https://arxiv.org/abs/2603.02250>`_.
