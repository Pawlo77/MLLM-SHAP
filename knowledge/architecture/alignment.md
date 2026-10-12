---
title: "alignment"
type: module
domain: alignment
parent: "[[mllm-shap]]"
tags: [architecture, alignment]
created: 2026-10-10
updated: 2026-10-11
related-code:
  - "[[src/mllm_shap/alignment/__init__.py]]"
  - "[[src/mllm_shap/alignment/ctc.py]]"
  - "[[src/mllm_shap/alignment/io.py]]"
  - "[[src/mllm_shap/alignment/partitions.py]]"
  - "[[src/mllm_shap/alignment/segments.py]]"
  - "[[src/mllm_shap/alignment/sgpa.py]]"
---

# alignment

Concept: [[SGPA]]. Decisions: [[Alignment decisions]].

## Responsibility

- Turn a transcript and its speech into one time segment per token, so audio segments can become players.
- The `sgpa_segments` partition turns segments into players. No backend and no audio rendering (that is [[masking]]); the `audio` extra (`librosa`, `soundfile`) plus the `transformers` extra for the default model.

## Public surface

- `SpectrogramGuidedAligner(model=None, energy_weight=0.8, flux_weight=0.2, separator="|")`; `align(transcript, audio: AudioPart, attach_audio=False) -> list[AudioSegment]`.
- `CtcModel` protocol (`sample_rate`, `blank_id`, `vocab`, `log_probs(waveform)`); `Wav2Vec2Ctc(model_id="facebook/wav2vec2-base-960h", revision=None, device="cpu")` implements it.
- `SgpaSegmentPartitioner` (`sgpa_segments`, options `SgpaOptions`: `model_id`, `revision`, `device`, `energy_weight`, `flux_weight`, `text_roles` for word players of typed messages): one player per spellable word of every message with an `AudioPart` and a transcript; ids `"{snapshot_id}:{turn}:{message}:{token}"`; spans carry the word's text range and `samples`; the default aligner loads on the first `partition`.
- `AudioSegment(token, token_index, start_time, end_time, ctc_start_time, ctc_end_time, confidence, boundary_refined, sample_rate, start_sample, end_sample, audio=b"")` with `duration`.
- `decode_audio(data, sample_rate=None)` (mono float32, optional resampling) and `encode_wav(waveform, sample_rate)` (16-bit PCM).
- `ctc.forced_align(log_probs, targets, blank)` and `ctc.target_spans(...)`: Viterbi alignment over the blank-extended target sequence.

## Data flow

- In: `AudioPart` from [[domain]] and a transcript string or token list.
- Steps: decode and resample for the model; `log_probs`; CTC forced alignment of the spellable characters (separator between tokens); token spans from their characters; each edge moves to the minimum of `energy_weight` × RMS (root mean square energy) + `flux_weight` × spectral flux within its search window when that point is quiet.
- Out: `AudioSegment`s in token order, at the source sample rate; `sgpa_segments` turns them into players for the [[pipeline]].

## Invariants

- Adjacent segments share one boundary, which lies inside the gap between their character spans.
- A token without a character in the model vocabulary gets no segment; a transcript with none raises `ValueError`.
- Segments last at least `MIN_DURATION_S` (0.05 s) and have `end_sample > start_sample`.
- The aligner changes no global state (warnings filters, library log levels).

## Known issues

- None open.
- [[D06 - SGPA paper not in the papers list]]: the method's paper is under review, so it has no [[Papers]] section yet.

## Tests

- [[tests/alignment/test_ctc.py]]
- [[tests/alignment/test_io.py]]
- [[tests/alignment/test_segments.py]]
- [[tests/alignment/test_sgpa.py]]
- [[tests/alignment/test_sgpa_players.py]] (also `mock_audio` end to end)
