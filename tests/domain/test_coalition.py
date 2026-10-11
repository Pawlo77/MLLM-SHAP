"""Tests for coalition masks.

@see [[src/mllm_shap/domain/coalition.py]]
"""

import pytest

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet


def test_coalition_mask_counts_present_players() -> None:
    """Coalition size counts True entries."""
    mask = CoalitionMask.from_sequence((True, False, True))
    assert mask.coalition_size() == 2


def test_coalition_mask_validates_against_player_set() -> None:
    """Mask length must match the player set."""
    players = PlayerSet(player_ids=("a", "b", "c"))
    mask = CoalitionMask.from_sequence((True, False))
    with pytest.raises(ValueError, match="does not match player count"):
        mask.validate_against(players)


def test_coalition_mask_rejects_empty_mask() -> None:
    """Coalition masks require at least one entry."""
    with pytest.raises(ValueError, match="at least one player"):
        CoalitionMask.from_sequence(())
