"""Tests for embedding-based value functions.

@see [[src/mllm_shap/value/embedding.py]]
"""

import pytest

from mllm_shap.domain.enums import EmbeddingMode
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.value.embedding import CosineEmbeddingValue, EuclideanEmbeddingValue


def _record(
    embedding: tuple[float, ...],
    contextual_embedding: tuple[float, ...] = (),
) -> GenerationRecord:
    return GenerationRecord(
        text="response",
        embedding=embedding,
        contextual_embedding=contextual_embedding,
    )


def test_cosine_embedding_u1_static_ranking() -> None:
    """Static embeddings preserve U1 ranking: identical > orthogonal > opposite."""
    base = _record((1.0, 0.0))
    identical = _record((1.0, 0.0))
    orthogonal = _record((0.0, 1.0))
    opposite = _record((-1.0, 0.0))
    value_fn = CosineEmbeddingValue(embedding_mode=EmbeddingMode.STATIC)

    identical_score = value_fn.score(base, identical)
    orthogonal_score = value_fn.score(base, orthogonal)
    opposite_score = value_fn.score(base, opposite)

    assert identical_score == pytest.approx(1.0, abs=1e-6)
    assert identical_score > orthogonal_score
    assert orthogonal_score > opposite_score


def test_cosine_embedding_u2_contextual_mode() -> None:
    """Contextual mode prefers contextual embeddings over static ones."""
    base = _record(
        embedding=(0.0, 1.0),
        contextual_embedding=(1.0, 0.0),
    )
    candidate = _record(
        embedding=(0.0, 1.0),
        contextual_embedding=(1.0, 0.0),
    )
    value_fn = CosineEmbeddingValue(embedding_mode=EmbeddingMode.CONTEXTUAL)
    assert value_fn.score(base, candidate) == pytest.approx(1.0, abs=1e-6)


def test_euclidean_embedding_u4_ranking() -> None:
    """Euclidean similarity decreases with embedding distance."""
    base = _record((0.0, 0.0))
    near = _record((0.1, 0.0))
    far = _record((3.0, 0.0))
    value_fn = EuclideanEmbeddingValue()

    near_score = value_fn.score(base, near)
    far_score = value_fn.score(base, far)
    assert near_score > far_score
    assert value_fn.score(base, base) == pytest.approx(1.0, abs=1e-6)


def test_embedding_value_requires_vector_or_provider() -> None:
    """Missing embeddings raise a type error."""
    value_fn = CosineEmbeddingValue()
    empty = GenerationRecord(text="missing")
    with pytest.raises(TypeError, match="EmbeddingProvider"):
        value_fn.score(empty, empty)
