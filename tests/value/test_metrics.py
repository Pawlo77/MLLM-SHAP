"""Tests for numpy similarity metrics.

@see [[src/mllm_shap/value/_metrics.py]]
"""

from mllm_shap.value._metrics import cosine_similarity


def test_cosine_similarity_identical_vectors() -> None:
    """Identical vectors score one."""
    base = (1.0, 0.0)
    assert cosine_similarity(base, base) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Orthogonal vectors score zero."""
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0
