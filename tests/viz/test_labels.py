"""Tests for attribution label helpers.

@see [[src/nlp_shap/viz/labels.py]]
"""

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.masking.partitions import TokenPartitioner
from nlp_shap.viz.labels import token_labels


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def test_token_labels_align_with_partition() -> None:
    """Token labels follow the same order as the token partitioner."""
    snapshot = _snapshot("refund within thirty days")
    player_set = TokenPartitioner().partition(snapshot)
    labels = token_labels(snapshot, player_set)
    assert labels == ("refund", "within", "thirty", "days")


def test_token_labels_reject_mismatched_player_set() -> None:
    """Misaligned player ids raise ValueError."""
    snapshot = _snapshot("one two")
    player_set = TokenPartitioner().partition(_snapshot("other text"))
    with pytest.raises(ValueError, match="does not align"):
        token_labels(snapshot, player_set)


def test_token_labels_prefix_the_speaker() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hi there", speaker="bob"),)),
    ))
    player_set = TokenPartitioner().partition(snapshot)
    assert token_labels(snapshot, player_set) == ("bob: hi", "bob: there")
