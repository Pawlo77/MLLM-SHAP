"""Tests for embedding-based value functions.

@see [[src/mllm_shap/value/embedding.py]]
"""

import numpy as np
import pytest

from mllm_shap.domain.enums import EmbeddingMode
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import PluginNotFoundError
from mllm_shap.plugins import PluginGroup, default_registry
from mllm_shap.value.embedding import CosineEmbeddingValue


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


def test_embedding_value_requires_vector_or_provider() -> None:
    """Missing embeddings raise a type error."""
    value_fn = CosineEmbeddingValue()
    empty = GenerationRecord(text="missing")
    with pytest.raises(TypeError, match="EmbeddingProvider"):
        value_fn.score(empty, empty)


class _AxisEncoder:
    """Sentence encoder that puts word counts of two keywords on two axes."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([[t.count("cat"), t.count("dog")] for t in texts], float)


def test_sentence_encoder_scores_the_text_not_backend_vectors() -> None:
    encoder = _AxisEncoder()
    value_fn = CosineEmbeddingValue(encoder=encoder)
    value_fn._encoder_name = "axis"
    base = GenerationRecord(text="cat cat dog", embedding=(1.0, 0.0))
    near = GenerationRecord(text="cat dog", embedding=(0.0, 1.0))
    far = GenerationRecord(text="dog dog dog")

    assert not value_fn.needs_embeddings
    assert value_fn.score(base, near) == pytest.approx(0.9487, abs=1e-4)
    assert value_fn.score(base, far) < value_fn.score(base, near)
    assert value_fn.score(base, base) == pytest.approx(1.0)
    calls = encoder.calls
    value_fn.score(base, near)
    assert encoder.calls == calls  # texts are encoded once


def test_default_encoder_is_a_sentence_model_and_hash_vectors_are_opt_in() -> None:
    from mllm_shap.pipeline.config import ExplainConfig
    from mllm_shap.plugins import PluginContext

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "m"},
        "explanation": {"value_fn": "embedding_cosine"},
    })
    value_fn = default_registry().resolve(
        PluginGroup.VALUE_FNS, "embedding_cosine", PluginContext(config=config)
    )
    assert not value_fn.needs_embeddings
    assert value_fn.manifest_info()["encoder"] == "all-MiniLM-L6-v2"
    assert value_fn.allow_hash_embeddings is False


def test_embedding_euclidean_is_removed_with_a_pointer_to_cosine() -> None:
    with pytest.raises(PluginNotFoundError, match="use 'embedding_cosine'"):
        default_registry().resolve_factory(PluginGroup.VALUE_FNS, "embedding_euclidean")
