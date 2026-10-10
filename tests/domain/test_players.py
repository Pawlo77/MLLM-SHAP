"""Tests for player sets.

@see [[src/nlp_shap/domain/players.py]]
"""

import pytest

from nlp_shap.domain.players import PlayerSet, PlayerSpan


def test_player_set_exposes_count() -> None:
    """Player sets report the number of registered players."""
    players = PlayerSet(player_ids=("t0", "t1", "t2"))
    assert players.num_players == 3


def test_player_set_rejects_empty_ids() -> None:
    """Player sets require at least one player."""
    with pytest.raises(ValueError, match="at least one player"):
        PlayerSet(player_ids=())


def test_player_set_rejects_duplicate_ids() -> None:
    """Player ids must be unique."""
    with pytest.raises(ValueError, match="unique"):
        PlayerSet(player_ids=("t0", "t0"))


def test_spans_carry_no_sample_range_unless_audio() -> None:
    text = PlayerSpan(message_index=0, start=0, end=2, text="hi")
    audio = PlayerSpan(message_index=0, start=0, end=2, text="hi", samples=(10, 90))
    assert text.samples is None
    assert audio.samples == (10, 90)
