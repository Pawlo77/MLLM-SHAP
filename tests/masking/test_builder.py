"""Tests for masked snapshot views and structural sharing.

@see [[src/nlp_shap/masking/builder.py]]
"""

import random

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.conversation import ConversationSnapshot
from nlp_shap.masking.builder import MaskBuilder
from nlp_shap.masking.partitions import TokenPartitioner
from nlp_shap.masking.policies import DeletePolicy


def test_masked_views_share_base_snapshot_id(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Many coalition views reference the same base snapshot identifier."""
    builder = MaskBuilder(DeletePolicy())
    players = TokenPartitioner().partition(sample_snapshot)
    rng = random.Random(42)
    views = []
    for _ in range(1000):
        present = tuple(rng.choice((True, False)) for _ in range(players.num_players))
        if not any(present):
            present = (True, *present[1:])
        mask = CoalitionMask.from_sequence(present)
        views.append(builder.view(sample_snapshot, players, mask))

    assert len({view.base_snapshot_id for view in views}) == 1
    assert views[0].base_snapshot_id == sample_snapshot.snapshot_id
    assert all(view.base is sample_snapshot for view in views)


def test_render_does_not_copy_base_snapshot(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Rendered snapshots are new objects while the base snapshot is reused."""
    builder = MaskBuilder(DeletePolicy())
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    view = builder.view(sample_snapshot, players, mask)
    rendered = builder.render(view)
    assert rendered is not sample_snapshot
    assert view.base is sample_snapshot
