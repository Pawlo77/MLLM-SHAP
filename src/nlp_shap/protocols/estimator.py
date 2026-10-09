"""Estimator strategy protocols.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from .estimand import EstimandAggregator


class EstimatorStrategy(Protocol):
    """Sample coalitions and turn their payoffs into attributions."""

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield coalition masks to evaluate for the player set."""

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Aggregate evaluated coalitions into per-player attributions."""


@runtime_checkable
class TwoPhaseEstimator(Protocol):
    """Estimator that allocates a second sampling phase from first-phase payoffs."""

    def begin_allocation(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
    ) -> None:
        """Consume phase-one payoffs and plan phase two."""

    def sample_allocation_masks(self) -> Iterator[CoalitionMask]:
        """Yield phase-two coalition masks."""
