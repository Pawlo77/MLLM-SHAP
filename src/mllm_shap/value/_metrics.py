"""Numpy similarity metrics for value functions.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_metrics.py]]
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(
    base: Sequence[float] | NDArray[np.floating],
    other: Sequence[float] | NDArray[np.floating],
) -> float:
    """Return cosine similarity between two vectors."""
    base_array = np.asarray(base, dtype=np.float64)
    other_array = np.asarray(other, dtype=np.float64)
    epsilon = 1e-8
    base_norm = max(float(np.linalg.norm(base_array)), epsilon)
    other_norm = max(float(np.linalg.norm(other_array)), epsilon)
    return float(np.dot(base_array, other_array) / (base_norm * other_norm))
