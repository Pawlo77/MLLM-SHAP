"""Tests for estimand domain types.

@see [[src/nlp_shap/domain/estimands.py]]
"""

import pytest

from nlp_shap.domain.estimands import Estimand, estimand_to_wire


def test_estimand_values_are_stable_strings() -> None:
    """Enum members serialize to the config and manifest wire values."""
    assert Estimand.SHAPLEY.value == "shapley"
    assert Estimand.BANZHAF.value == "banzhaf"


def test_estimand_from_value_round_trip() -> None:
    """Wire values map back to enum members."""
    assert Estimand("shapley") is Estimand.SHAPLEY
    assert Estimand("banzhaf") is Estimand.BANZHAF


def test_estimand_rejects_unknown_value() -> None:
    """Unknown wire values raise a lookup error."""
    with pytest.raises(ValueError, match="not a valid"):
        Estimand("marginal")


def test_estimand_to_wire_matches_enum_values() -> None:
    """Wire serialization preserves estimand labels."""
    assert estimand_to_wire(Estimand.SHAPLEY) == "shapley"
    assert estimand_to_wire(Estimand.BANZHAF) == "banzhaf"
