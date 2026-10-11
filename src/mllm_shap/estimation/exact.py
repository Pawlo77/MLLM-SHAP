"""Exact coalition enumeration estimator.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_exact.py]]
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ._shared import _SnapshotEstimator, mask_int_to_present

if TYPE_CHECKING:
    pass


class ExactEstimator(_SnapshotEstimator):
    """Enumerate all coalitions and delegate attribution to an estimand plugin."""

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "exact"

    @staticmethod
    def num_coalitions(num_players: int) -> int:
        """Return the number of coalitions evaluated by exact enumeration (2^n)."""
        if num_players < 1:
            msg = "num_players must be at least 1"
            raise ValueError(msg)
        return 1 << num_players

    @staticmethod
    def iter_mask_ints(num_players: int) -> Iterator[int]:
        """Lazily yield every coalition bitmask from empty to grand."""
        yield from range(1 << num_players)

    mask_int_to_present = staticmethod(mask_int_to_present)

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield every coalition mask including the empty and grand coalitions."""
        if budget_fraction != 1.0:
            msg = "exact estimator requires budget_fraction == 1.0"
            raise ValueError(msg)
        _ = include_minimal_masks, seed
        return self.iter_masks(player_set)

    @staticmethod
    def iter_masks(player_set: PlayerSet) -> Iterator[CoalitionMask]:
        """Yield every coalition mask including the empty and grand coalitions."""
        num_players = player_set.num_players
        for mask_int in ExactEstimator.iter_mask_ints(num_players):
            present = mask_int_to_present(mask_int, num_players)
            yield CoalitionMask.from_sequence(present)
