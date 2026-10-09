"""Tests for numpy similarity metrics.

@see [[src/nlp_shap/value/_metrics.py]]
"""

import numpy as np

from nlp_shap.value._metrics import cosine_similarity, euclidean_similarity


def test_cosine_similarity_identical_vectors() -> None:
    """Identical vectors score one."""
    base = (1.0, 0.0)
    assert cosine_similarity(base, base) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Orthogonal vectors score zero."""
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_euclidean_similarity_monotonic_with_distance() -> None:
    """Closer vectors receive higher similarity."""
    base = (0.0, 0.0, 0.0)
    near = (0.1, 0.0, 0.0)
    far = (3.0, 0.0, 0.0)
    assert euclidean_similarity(base, near) > euclidean_similarity(base, far)


def test_euclidean_similarity_identical_vectors() -> None:
    """Identical vectors score one."""
    base = np.array([1.0, 2.0], dtype=np.float64)
    assert euclidean_similarity(base, base) == 1.0
