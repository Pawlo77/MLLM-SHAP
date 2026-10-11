"""Estimand aggregation protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Sequence
from typing import Protocol

from ..domain.estimands import Estimand


class EstimandAggregator(Protocol):
    """Aggregate coalition payoffs into per-player attributions."""

    @property
    def estimand(self) -> Estimand:
        """Return the estimand label produced by this aggregator."""

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return coalition weight for the given size excluding the focal player."""

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into per-player attribution values."""
