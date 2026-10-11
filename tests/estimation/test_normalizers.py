"""Tests for attribution normalizers.

@see [[src/mllm_shap/estimation/normalizers.py]]
"""

import numpy as np
import pytest

from mllm_shap.estimation.normalizers import (
    AbsSumNormalizer,
    IdentityNormalizer,
    MinMaxNormalizer,
    PowerShiftNormalizer,
)


@pytest.fixture
def sample_values() -> tuple[float, ...]:
    """Sample attribution vector."""
    return (1.0, -2.0, 3.0)


def test_identity_normalizer_returns_copy(sample_values: tuple[float, ...]) -> None:
    """Identity normalizer returns equivalent values."""
    normalizer = IdentityNormalizer()
    assert normalizer.normalize(sample_values) == list(sample_values)


def test_abs_sum_normalizer_scales_to_unit_l1(
    sample_values: tuple[float, ...],
) -> None:
    """Absolute values sum to one after normalization."""
    normalizer = AbsSumNormalizer()
    normalized = normalizer.normalize(sample_values)
    assert np.isclose(sum(abs(value) for value in normalized), 1.0)


def test_abs_sum_normalizer_preserves_zeros() -> None:
    """Zero entries remain zero."""
    normalizer = AbsSumNormalizer()
    normalized = normalizer.normalize((0.0, 2.0, -3.0, 0.0))
    assert normalized[0] == 0.0
    assert normalized[3] == 0.0


def test_power_shift_normalizer_sums_to_one(sample_values: tuple[float, ...]) -> None:
    """Power-shift output is non-negative and sums to one."""
    normalizer = PowerShiftNormalizer(power=2.0)
    normalized = normalizer.normalize(sample_values)
    assert all(value >= 0.0 for value in normalized)
    assert np.isclose(sum(normalized), 1.0)


def test_power_shift_normalizer_rejects_non_positive_power() -> None:
    """Invalid power values raise ValueError."""
    with pytest.raises(ValueError, match="positive"):
        PowerShiftNormalizer(power=0.0)


def test_min_max_normalizer_maps_to_unit_interval(
    sample_values: tuple[float, ...],
) -> None:
    """Min-max output lies in [0, 1] and sums to one."""
    normalizer = MinMaxNormalizer()
    normalized = normalizer.normalize(sample_values)
    assert all(0.0 <= value <= 1.0 for value in normalized)
    assert np.isclose(sum(normalized), 1.0)


def test_min_max_normalizer_uniform_for_equal_values() -> None:
    """Equal attributions become a uniform distribution."""
    normalizer = MinMaxNormalizer()
    normalized = normalizer.normalize((3.0, 3.0, 3.0))
    assert normalized == pytest.approx([1 / 3, 1 / 3, 1 / 3])
