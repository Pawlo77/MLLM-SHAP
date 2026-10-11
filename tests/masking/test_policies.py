"""Tests for absence-policy rendering.

@see [[src/mllm_shap/masking/policies.py]]
"""

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.conversation import ConversationSnapshot
from mllm_shap.masking.builder import MaskBuilder
from mllm_shap.masking.partitions import TokenPartitioner
from mllm_shap.masking.policies import DeletePolicy, NeutralPolicy, PadPolicy


def test_delete_policy_removes_absent_tokens(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Absent tokens are omitted from rendered text."""
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    rendered = DeletePolicy().apply(sample_snapshot, players, mask)
    assert rendered.turns[0].messages[0].text == "Who you?"
    assert rendered.turns[0].messages[1].text == "I am helpful."


def test_pad_policy_uses_placeholder(sample_snapshot: ConversationSnapshot) -> None:
    """Absent tokens are replaced with the mask placeholder."""
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    rendered = PadPolicy().apply(sample_snapshot, players, mask)
    assert rendered.turns[0].messages[0].text == "Who [MASK] you?"


def test_neutral_policy_preserves_token_width(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Absent tokens are replaced with width-matched neutral fillers."""
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    rendered = NeutralPolicy().apply(sample_snapshot, players, mask)
    assert rendered.turns[0].messages[0].text == "Who ___ you?"


def test_delete_policy_empty_coalition_yields_empty_text(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Empty coalition keeps structure with empty message text for v(∅)."""
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((False,) * players.num_players)
    rendered = DeletePolicy().apply(sample_snapshot, players, mask)
    assert all(
        message.text == "" for turn in rendered.turns for message in turn.messages
    )


def test_mask_builder_renders_via_policy(sample_snapshot: ConversationSnapshot) -> None:
    """MaskBuilder materializes coalition-specific snapshots."""
    policy = PadPolicy(placeholder="<mask>")
    builder = MaskBuilder(policy)
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    view = builder.view(sample_snapshot, players, mask)
    rendered = builder.render(view)
    assert rendered.turns[0].messages[0].text == "Who <mask> you?"
