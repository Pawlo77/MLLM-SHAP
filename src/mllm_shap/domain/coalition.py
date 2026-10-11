"""Coalition masks over explainability players.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_coalition.py]]
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from .players import PlayerSet


@dataclass(frozen=True, slots=True)
class CoalitionMask:
    """Boolean presence mask aligned with a player set."""

    present: tuple[bool, ...]
    """Presence flags in :class:`~mllm_shap.domain.players.PlayerSet` order."""

    def __post_init__(self) -> None:
        """Reject empty masks."""
        if not self.present:
            msg = "coalition mask must include at least one player"
            raise ValueError(msg)

    @classmethod
    def from_sequence(cls, present: Sequence[bool]) -> Self:
        """Build a mask from any boolean sequence."""
        return cls(present=tuple(present))

    def coalition_size(self) -> int:
        """Return the number of present players."""
        return sum(self.present)

    def validate_against(self, player_set: PlayerSet) -> None:
        """Ensure the mask length matches the player set."""
        if len(self.present) != player_set.num_players:
            msg = (
                "coalition mask length "
                f"{len(self.present)} does not match player count "
                f"{player_set.num_players}"
            )
            raise ValueError(msg)
