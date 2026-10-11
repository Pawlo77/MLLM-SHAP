"""Masking and absence policy protocols.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from typing import Protocol

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet


class AbsencePolicy(Protocol):
    """Render absent players when building a masked snapshot view."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Return a masked snapshot view for coalition evaluation."""
