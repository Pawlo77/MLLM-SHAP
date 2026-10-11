"""Partitioners own player spans; policies and labels never re-tokenise.

@see [[knowledge/architecture/masking]]
"""

import re

import pytest

from mllm_shap import ExplainConfig, ExplainRunner, PluginGroup
from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.players import PlayerSet, PlayerSpan
from mllm_shap.masking.partitions import TokenPartitioner
from mllm_shap.masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from mllm_shap.plugins import default_registry
from mllm_shap.viz.labels import token_labels


class PairPartitioner:
    """Group every two whitespace tokens of each message into one player."""

    def __init__(self, context: object | None = None) -> None:
        _ = context

    @property
    def name(self) -> str:
        return "pairs"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        spans: list[PlayerSpan] = []
        messages = [m for turn in snapshot.turns for m in turn.messages]
        for message_index, message in enumerate(messages):
            tokens = list(re.finditer(r"\S+", message.text))
            for first in range(0, len(tokens), 2):
                group = tokens[first : first + 2]
                start, end = group[0].start(), group[-1].end()
                spans.append(
                    PlayerSpan(message_index, start, end, message.text[start:end])
                )
        ids = tuple(f"{snapshot.snapshot_id}:pair:{i}" for i in range(len(spans)))
        return PlayerSet(player_ids=ids, spans=tuple(spans))


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="be brief"),
                Message(role=Role.USER, text="one two three four five"),
            )
        ),
    ))


def test_token_partitioner_fills_spans() -> None:
    players = TokenPartitioner().partition(_snapshot())
    assert len(players.spans) == players.num_players == 7
    assert players.spans[2] == PlayerSpan(1, 0, 3, "one")
    assert players.spans[-1] == PlayerSpan(1, 19, 23, "five")


def test_player_set_rejects_span_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one span per player"):
        PlayerSet(player_ids=("a", "b"), spans=(PlayerSpan(0, 0, 1, "x"),))


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (DeletePolicy(), "five"),
        (PadPolicy(), "[MASK] [MASK] five"),
        (NeutralPolicy(), "_______ __________ five"),
    ],
)
def test_policies_render_custom_spans(policy: object, expected: str) -> None:
    snapshot = _snapshot()
    players = PairPartitioner().partition(snapshot)
    mask = CoalitionMask.from_sequence((True, False, False, True))
    rendered = policy.apply(snapshot, players, mask)
    assert rendered.turns[0].messages[0].text == "be brief"
    assert rendered.turns[0].messages[1].text == expected


def test_policies_keep_text_outside_spans() -> None:
    """Text no player covers is rendered unchanged."""
    snapshot = _snapshot()
    players = TokenPartitioner().partition(snapshot)
    only_user = PlayerSet(player_ids=players.player_ids[2:], spans=players.spans[2:])
    mask = CoalitionMask.from_sequence((False,) * only_user.num_players)
    rendered = DeletePolicy().apply(snapshot, only_user, mask)
    assert rendered.turns[0].messages[0].text == "be brief"
    assert rendered.turns[0].messages[1].text == ""


def test_policies_require_spans() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="spans"):
        DeletePolicy().apply(
            snapshot,
            PlayerSet(player_ids=("a",)),
            CoalitionMask.from_sequence((True,)),
        )


def test_labels_come_from_spans() -> None:
    snapshot = _snapshot()
    assert token_labels(snapshot, PairPartitioner().partition(snapshot)) == (
        "be brief",
        "one two",
        "three four",
        "five",
    )


@pytest.mark.parametrize("absence_policy", ["delete", "pad", "neutral"])
def test_custom_partitioner_runs_end_to_end(absence_policy: str) -> None:
    registry = default_registry()
    registry.register(PluginGroup.PARTITIONS, "pairs", PairPartitioner)
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "players": "pairs",
            "absence_policy": absence_policy,
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config, registry=registry).explain_sync(_snapshot())
    assert len(output.result.values) == 4
    assert output.metrics.requested == 16
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )
