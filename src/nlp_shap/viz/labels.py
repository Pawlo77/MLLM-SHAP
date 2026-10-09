"""Token label helpers for attribution renderers.

@domain viz
@see [[knowledge/architecture/viz]]
@tested-by [[tests/viz/test_labels.py]]
"""

from collections.abc import Sequence

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet


def token_labels(
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
) -> tuple[str, ...]:
    """Return every player's ``PlayerSpan.display_label`` in mask order.

    Raises:
        ValueError: If the player set has no spans or its spans do not match
            ``snapshot``.
    """
    messages = snapshot.messages()
    if not player_set.spans:
        msg = "player set has no spans; build it with a partitioner"
        raise ValueError(msg)
    for span in player_set.spans:
        if (
            span.message_index >= len(messages)
            or messages[span.message_index].text[span.start : span.end] != span.text
        ):
            msg = "player set does not align with the snapshot"
            raise ValueError(msg)
    return tuple(span.display_label for span in player_set.spans)


def check_attribution(labels: Sequence[str], values: Sequence[float]) -> None:
    """Raise ``ValueError`` unless ``labels`` and ``values`` are equal and non-empty."""
    if len(labels) != len(values):
        msg = "labels and values must have the same length"
        raise ValueError(msg)
    if not labels:
        msg = "cannot render an empty attribution"
        raise ValueError(msg)
