---
title: "value"
type: module
domain: value
parent: "[[mllm-shap]]"
tags: [architecture, value]
created: 2026-10-07
updated: 2026-10-18
related-code:
  - "[[src/mllm_shap/value/__init__.py]]"
  - "[[src/mllm_shap/value/_metrics.py]]"
  - "[[src/mllm_shap/value/_token_hash.py]]"
  - "[[src/mllm_shap/value/bertscore.py]]"
  - "[[src/mllm_shap/value/contrastive.py]]"
  - "[[src/mllm_shap/value/embedding.py]]"
  - "[[src/mllm_shap/value/encoders.py]]"
  - "[[src/mllm_shap/value/generation_records.py]]"
  - "[[src/mllm_shap/value/judge.py]]"
  - "[[src/mllm_shap/value/logprob.py]]"
  - "[[src/mllm_shap/value/nli.py]]"
  - "[[src/mllm_shap/value/semascore.py]]"
  - "[[src/mllm_shap/value/speech.py]]"
  - "[[src/mllm_shap/value/teacher_forced.py]]"
  - "[[src/mllm_shap/value/tfidf.py]]"
  - "[[src/mllm_shap/value/wer.py]]"
---
# value

## Responsibility

- Score one coalition generation against the [[Base generation]]; the score is the coalition payoff `v(S)` (Value function).
- Build the generation records both `explain` and `reanalyze` score.

## Public surface

- Every built-in declares `info: ValueInfo` (range, `reads_base`, `needs`, `extra_model`, `cost`); `tests/value/test_info.py` checks it and `docs/theory/value_functions.rst` lists it.
- `TfIdfCosineValue` (`tfidf_cosine`, default): a `FittableValueFunction`; `fit(corpus)` freezes inverse document frequencies (IDF) over the base and all coalition records; `score` is the term frequency-inverse document frequency (TF-IDF) cosine of `text_token_rows`.
- `CosineEmbeddingValue` (`embedding_cosine`, options `encoder`, `allow_hash_embeddings`): cosine of sentence-transformers embeddings of the answers (default `all-MiniLM-L6-v2`); `encoder: model_states` reads the backend's `embedding` or `contextual_embedding` (`needs_embeddings`). `embedding_euclidean` is removed.
- `SelfLogprobValue` (`self_logprob`): mean of the candidate's `logprobs`, 0 when empty, ignores the base; `LogprobValue` (`logprob`) is its deprecated alias ([[Value decisions#2026-10-18 — Teacher-forced value functions score y_N without generating|decision]]).
- `ReferenceWerValue` (`reference_wer`, option `reference`): `1 - min(1, WER)`, word-level Levenshtein over lowercased `[\w']+` words.
- `TfLogprobValue` (`tf_logprob`), `TfJsdValue` (`tf_jsd`, option `top_k`) and `ContrastiveValue` (`contrastive`, options `target`, `foil`) implement `RequiresTeacherForcing`: they read `GenerationRecord.forced[y]` for `forced_continuations(base)` and, with `reads_candidate_text` false, no coalition answer is generated. `tf_jsd` is `-sum_t JSD` over each position's top-`k` law plus a rest bucket.
- `BertScoreValue`, `NliLogOddsValue`, `SemaScoreValue`: `fit(corpus)` batch-encodes the answers; encoders load lazily from `encoders.py` (`TokenEncoder`, `NliModel`, `WordEncoder`) and tests inject fakes.
- `SpeechBertScoreValue` reads `audio_path` frames; `TranscribedValue` (`transcribed`) transcribes then applies a wrapped text value function.
- `JudgeGradeValue` (`judge_grade`): `PreparesScores.prepare` grades each distinct answer with a judge backend (digit probabilities via teacher forcing, else `samples` generations); refuses more than `max_coalitions` answers.
- `manifest_info()` (optional) feeds `RunManifest.value_info`.
- `generation_record_from_text(text)`: token rows from `text.split()` with 56-bit `blake2b` ids and 8-dimensional hash embeddings, all derived from the text only.

## Data flow

- In: records from `generation_record_from_text`, built by the [[pipeline]] orchestrator from backend text (with logprobs, model embeddings or `forced` scores when the value function needs them) and by `reanalyze` and `rescore` from the archive.
- Out: one float per coalition. Imports [[domain]], [[protocols]], [[plugins]] (context type only).

## Invariants

- `explain` and `reanalyze` return the same values on the same archive for text value functions, `self_logprob`, sentence-encoder `embedding_cosine` and archived teacher-forced scores.
- TF-IDF weights are fitted once per scoring pass on exactly the corpus that is scored.
- `score(base, base)` is the grand-coalition payoff: 1 for `tfidf_cosine`, 0 for `tf_jsd`.

## Known issues

- Embedding values of a transformers run with `encoder: model_states` differ between `explain` (model embeddings) and `reanalyze` (hash embeddings of archived text).
- Speech value functions need records with `audio_path`; no built-in backend returns speech.
- Fixed: C08, C15 ([[Fixed issues]]).

## Tests

- [[tests/value/test_tfidf.py]]
- [[tests/value/test_embedding.py]]
- [[tests/value/test_wer.py]]
- [[tests/value/test_logprob.py]]
- [[tests/value/test_teacher_forced.py]]
- [[tests/value/test_semantic.py]]
- [[tests/value/test_speech.py]]
- [[tests/value/test_judge.py]]
- [[tests/value/test_info.py]]
- [[tests/value/test_metrics.py]]
- [[tests/value/test_generation_records.py]]
- [[tests/benchmarks/test_value_bench.py]]
