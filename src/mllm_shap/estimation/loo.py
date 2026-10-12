"""Leave-one-out attribution: each player's drop from the full coalition.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_loo.py]]
"""

from collections.abc import Iterator, Sequence

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from ._shared import _SnapshotEstimator


class LeaveOneOutEstimator(_SnapshotEstimator):
    """Attribute ``v(N) - v(N - {i})`` to player ``i`` from ``n + 1`` coalitions.

    The baseline every attribution method should beat. It ignores the budget and
    the estimand: values are not Shapley or Banzhaf values and need not add up
    to ``v(N) - v(empty)``.
    """

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "loo"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield the grand coalition, then the grand coalition without each player."""
        _ = budget_fraction, include_minimal_masks, seed
        n = player_set.num_players
        yield CoalitionMask.from_sequence((True,) * n)
        for i in range(n):
            yield CoalitionMask.from_sequence(tuple(j != i for j in range(n)))

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator | None,
    ) -> list[float]:
        """Return ``v(N) - v(N - {i})`` for every player ``i``."""
        _ = aggregator
        payoff = {
            mask.present: value for mask, value in zip(masks, payoffs, strict=True)
        }
        n = len(masks[0].present)
        grand = payoff[(True,) * n]
        return [grand - payoff[tuple(j != i for j in range(n))] for i in range(n)]
