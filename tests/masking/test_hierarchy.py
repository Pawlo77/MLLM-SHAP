"""Sentence players and the two-level grouping helpers.

@see [[src/mllm_shap/masking/hierarchy.py]]
"""

import pytest

from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.masking.hierarchy import (
    coarse_player_set,
    group_players,
    refined_player_set,
)
from mllm_shap.masking.partitions import SentencePartitioner
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import PluginContext

TEXT = "One two. Three four! Five?\n\nSix seven\n\nEight nine. Ten"


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.SYSTEM, text="Be brief."),)),
        Turn(messages=(Message(role=Role.USER, text=TEXT),)),
    ))


def test_sentences_end_at_punctuation_or_the_paragraph() -> None:
    players = SentencePartitioner().partition(_snapshot())
    assert [span.text for span in players.spans] == [
        "Be brief.",
        "One two.",
        "Three four!",
        "Five?",
        "Six seven",
        "Eight nine.",
        "Ten",
    ]
    assert all(
        _snapshot().messages()[s.message_index].text[s.start : s.end] == s.text
        for s in players.spans
    )
    assert len(set(players.player_ids)) == 7


def test_groups_are_paragraphs_or_messages() -> None:
    snapshot = _snapshot()
    players = SentencePartitioner().partition(snapshot)
    assert group_players(snapshot, players, "paragraph") == [
        [0],
        [1, 2, 3],
        [4],
        [5, 6],
    ]
    assert group_players(snapshot, players, "message") == [[0], [1, 2, 3, 4, 5, 6]]


def test_coarse_players_cover_their_members_and_refinement_keeps_the_rest() -> None:
    snapshot = _snapshot()
    players = SentencePartitioner().partition(snapshot)
    groups = group_players(snapshot, players, "paragraph")
    coarse = coarse_player_set(snapshot, players, groups)
    assert [span.text for span in coarse.spans] == [
        "Be brief.",
        "One two. Three four! Five?",
        "Six seven",
        "Eight nine. Ten",
    ]
    refined = refined_player_set(players, groups, [3, 1])
    assert [span.text for span in refined.spans] == [
        "One two.",
        "Three four!",
        "Five?",
        "Eight nine.",
        "Ten",
    ]


def test_sentences_need_an_explainable_role() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="Hi."),)),
    ))
    with pytest.raises(ValueError, match="at least one explainable sentence"):
        SentencePartitioner().partition(
            ConversationSnapshot.from_turns((
                Turn(messages=(Message(role=Role.USER, text="   "),)),
            ))
        )
    assert SentencePartitioner().partition(snapshot).num_players == 1


def test_trailing_whitespace_is_not_part_of_a_sentence() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="Go on.  Then stop   "),)),
    ))
    players = SentencePartitioner().partition(snapshot)
    assert [span.text for span in players.spans] == ["Go on.", "Then stop"]
    assert SentencePartitioner().name == "sentences"


def test_sentences_skip_roles_outside_the_option() -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
    })
    context = PluginContext(config=config, options={"roles": ["user"]})
    players = SentencePartitioner(context).partition(_snapshot())
    assert [span.text for span in players.spans][:2] == ["One two.", "Three four!"]
    assert len(players.spans) == 6
