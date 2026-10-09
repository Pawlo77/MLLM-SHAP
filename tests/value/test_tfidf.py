"""Tests for TF-IDF cosine value function.

@see [[src/nlp_shap/value/tfidf.py]]
"""

import math

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.value._token_hash import TokenHasher
from nlp_shap.value.tfidf import TfIdfCosineValue


def _record(
    text_tokens: tuple[tuple[int, ...], ...],
    audio_tokens: tuple[tuple[int, ...], ...] = (),
    text: str = "response",
) -> GenerationRecord:
    return GenerationRecord(
        text=text,
        text_token_rows=text_tokens,
        audio_token_rows=audio_tokens,
    )


def _sklearn_cosine(
    corpus: list[tuple[int, ...]],
    base_index: int,
    other_index: int,
) -> float:
    vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens)
    matrix = vectorizer.fit_transform(corpus).toarray()
    base = matrix[base_index]
    other = matrix[other_index]
    base_norm = max(float(np.linalg.norm(base)), 1e-8)
    other_norm = max(float(np.linalg.norm(other)), 1e-8)
    return float(np.dot(base, other) / (base_norm * other_norm))


def test_tfidf_identical_responses_score_one() -> None:
    """Identical token rows score one after fit."""
    base = _record(((1, 0), (0, 1)), ((0, 1), (1, 0)))
    value_fn = TfIdfCosineValue()
    value_fn.fit((base, base))
    score = value_fn.score(base, base)
    assert score == pytest.approx(1.0, abs=1e-6)


def test_tfidf_penalizes_token_changes() -> None:
    """Altered token rows score below the base response."""
    base = _record(((1, 0),), ((0, 1),))
    altered = _record(((1, 0), (0, 1)), ((0, 1),))
    value_fn = TfIdfCosineValue()
    value_fn.fit((base, altered))
    assert value_fn.score(base, base) == pytest.approx(1.0, abs=1e-6)
    assert value_fn.score(base, altered) < value_fn.score(base, base)


def test_tfidf_requires_fit_before_score() -> None:
    """Scoring before fit raises a runtime error."""
    base = _record(((1, 0),), ())
    value_fn = TfIdfCosineValue()
    with pytest.raises(RuntimeError, match="fit"):
        value_fn.score(base, base)


def test_tfidf_matches_sklearn_on_fixed_corpus() -> None:
    """Frozen-corpus scores match sklearn fit-transform on the same corpus."""
    base = _record(((1, 0), (0, 1)), ((0, 1), (1, 0)))
    altered = _record(((1, 0), (0, 1), (2, 3)), ((0, 1),))
    corpus = (base, altered)
    value_fn = TfIdfCosineValue()
    value_fn.fit(corpus)

    hashed_docs = [TokenHasher().document_token_ids(record) for record in corpus]
    expected = _sklearn_cosine(hashed_docs, 0, 1)
    actual = value_fn.score(base, altered)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_tfidf_frozen_idf_is_stable_across_scoring_order() -> None:
    """IDF weights do not change between score calls."""
    base = _record(((1, 0),), ())
    other = _record(((2, 3),), ())
    value_fn = TfIdfCosineValue()
    value_fn.fit((base, other))
    first = value_fn.score(base, other)
    second = value_fn.score(other, base)
    assert first == pytest.approx(second, abs=1e-6)
    assert math.isfinite(first)
