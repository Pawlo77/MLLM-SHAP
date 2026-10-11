"""Player partition protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from typing import Protocol

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet


class PlayerPartitioner(Protocol):
    """Partition a conversation snapshot into explainability players."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive ordered players and their spans from the snapshot.

        The returned :class:`~mllm_shap.domain.players.PlayerSet` must carry one
        :class:`~mllm_shap.domain.players.PlayerSpan` per player; absence policies
        and labels read the spans instead of re-tokenising.
        """
