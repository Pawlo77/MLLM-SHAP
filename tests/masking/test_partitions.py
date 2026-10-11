"""Tests for the token and message partition plugins.

@see [[src/mllm_shap/masking/partitions.py]]
"""

import pytest

from mllm_shap import ExplainConfig, ExplainRunner, PluginGroup
from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.masking.partitions import MessagePartitioner, TokenPartitioner
from mllm_shap.plugins import PluginContext, default_registry


def test_token_partitioner_splits_whitespace_tokens(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Whitespace tokens become ordered explainability players."""
    partitioner = TokenPartitioner()
    players = partitioner.partition(sample_snapshot)
    assert partitioner.name == "tokens"
    assert players.num_players == 6
    assert players.player_ids[0].startswith(sample_snapshot.snapshot_id)


def test_token_partitioner_rejects_empty_text() -> None:
    """Snapshots without explainable tokens are rejected."""
    turn = Turn(messages=(Message(role=Role.USER, text="   "),))
    snapshot = ConversationSnapshot.from_turns((turn,))
    with pytest.raises(ValueError, match="at least one explainable token"):
        TokenPartitioner().partition(snapshot)


def _three_messages() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="Answer in one word."),
                Message(role=Role.USER, text="Is the sky blue?", speaker="alice"),
                Message(role=Role.USER, text="And the grass?", speaker="bob"),
            )
        ),
    ))


def _messages_context(**options: object) -> PluginContext:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "explanation": {"players": "messages"},
    })
    return PluginContext(config=config, options=options)


def test_message_partitioner_makes_one_player_per_message() -> None:
    snapshot = _three_messages()
    players = (
        default_registry()
        .resolve(PluginGroup.PARTITIONS, "messages", _messages_context())
        .partition(snapshot)
    )

    assert players.player_ids == tuple(
        f"{snapshot.snapshot_id}:0:{i}" for i in range(3)
    )
    assert [span.text for span in players.spans] == [
        m.text for m in snapshot.messages()
    ]
    assert [span.display_label for span in players.spans] == [
        "system 1: Answer in one word.",
        "user/alice 2: Is the sky blue?",
        "user/bob 3: And the grass?",
    ]


def test_message_partitioner_truncates_labels_to_30_characters() -> None:
    text = "x" * 40
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))
    (span,) = MessagePartitioner().partition(snapshot).spans
    assert span.display_label == f"user 1: {'x' * 30}…"


def test_message_partitioner_roles_option_excludes_roles() -> None:
    players = MessagePartitioner(_messages_context(roles=["user"])).partition(
        _three_messages()
    )
    assert [span.message_index for span in players.spans] == [1, 2]


@pytest.mark.parametrize(
    ("policy", "middle"), [("delete", ""), ("pad", "[MASK]"), ("neutral", "_" * 16)]
)
def test_absence_policies_remove_the_absent_message(policy: str, middle: str) -> None:
    snapshot = _three_messages()
    players = MessagePartitioner().partition(snapshot)
    rendered = (
        default_registry()
        .resolve(PluginGroup.ABSENCE_POLICIES, policy, _messages_context())
        .apply(snapshot, players, CoalitionMask.from_sequence((True, False, True)))
    )

    assert [m.text for m in rendered.messages()] == [
        "Answer in one word.",
        middle,
        "And the grass?",
    ]
    assert rendered.messages()[1].speaker == "alice"


def test_exact_message_run_satisfies_efficiency() -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "explanation": {
            "estimator": "exact",
            "players": "messages",
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config).explain_sync(_three_messages())

    assert len(output.result.values) == 3
    assert output.metrics.requested == 8
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )
