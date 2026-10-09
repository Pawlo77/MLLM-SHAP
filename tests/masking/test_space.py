"""Tests for mask-space projection.

@see [[src/nlp_shap/masking/space.py]]
"""

import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.masking.space import MaskSpace


def test_materialize_projects_split_to_full_mask() -> None:
    """Explainable splits map back to the full coalition mask."""
    space = MaskSpace(
        explainable_mask=(False, True, True, False, True), target_length=5
    )
    split = CoalitionMask.from_sequence((True, False, True))
    assert space.materialize(split) == (True, True, False, True, True)


def test_materialize_rejects_mismatched_split_length() -> None:
    """Split length must match the explainable feature count."""
    space = MaskSpace(explainable_mask=(True, True, False), target_length=3)
    split = CoalitionMask.from_sequence((True,))
    with pytest.raises(ValueError, match="feature count"):
        space.materialize(split)


def test_mask_space_rejects_empty_explainable_mask() -> None:
    """At least one explainable position is required."""
    with pytest.raises(ValueError, match="at least one explainable"):
        MaskSpace(explainable_mask=(False, False), target_length=2)
