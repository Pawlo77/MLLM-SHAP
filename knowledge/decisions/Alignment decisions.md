---
title: "Alignment decisions"
type: decision-log
domain: alignment
parent: "[[mllm-shap]]"
created: 2026-10-10
updated: 2026-10-10
tags: [decisions, alignment]
related-code:
  - "[[src/mllm_shap/alignment/ctc.py]]"
  - "[[src/mllm_shap/alignment/sgpa.py]]"
---
# Alignment decisions

Architecture: [[alignment]]. Related: [[SGPA]].

## 2026-10-10 — Own CTC alignment instead of torchaudio

Decision: `alignment.ctc.forced_align` is a NumPy Viterbi over the blank-extended target sequence, resampling uses `librosa`, and the CTC model is a `CtcModel` protocol that `Wav2Vec2Ctc` implements. The `audio` extra holds only `librosa` and `soundfile`.

Why: The earlier port called `torchaudio.functional.forced_align`, which torchaudio deprecated for removal from its application programming interface (API), and torchaudio releases trail and pin torch versions (2.11 against our torch 2.13).

Consequences: Alignment, boundary refinement and audio I/O are tested in continuous integration with a known-truth fake model; only `Wav2Vec2Ctc` needs the `transformers` extra. Rejected: keeping torchaudio (deprecated API, version lock); `MagicMock`-patched tests of the model (they test the mock, not the alignment).
