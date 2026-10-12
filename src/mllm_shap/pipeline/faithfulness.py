"""Deletion curves: how fast a ranking of players destroys the answer.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_faithfulness.py]]
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..estimation._shared import _SnapshotEstimator
from ..plugins import PluginGroup, PluginRegistry, default_registry
from ..protocols.backend import GenerativeBackend
from ..protocols.estimand import EstimandAggregator
from .config import BudgetConfig, ExplainConfig
from .runner import ExplainRunner

_ESTIMATOR = "_deletion_curve"


@dataclass(frozen=True, slots=True)
class DeletionCurve:
    """Value of the coalition left after removing the top-``k`` players, per ``k``."""

    num_players: int
    """Number of players of the explained snapshot."""

    grand_value: float
    """``v(N)``, the value with every player present."""

    removed: tuple[int, ...]
    """Numbers ``k`` of top-ranked players removed, ascending."""

    values: tuple[float, ...]
    """``v(N - top_k)`` for each ``k`` in :attr:`removed`."""

    @property
    def drops(self) -> tuple[float, ...]:
        """Return ``v(N) - v(N - top_k)`` for each ``k``."""
        return tuple(self.grand_value - value for value in self.values)

    @property
    def area(self) -> float:
        """Return the trapezoid area of :attr:`values` over ``k / n``.

        Lower is more faithful: the value falls sooner when the ranking puts the
        players that matter first.
        """
        xs = [k / self.num_players for k in self.removed]
        pairs = zip(xs, xs[1:], self.values, self.values[1:], strict=False)
        return sum((x1 - x0) * (y0 + y1) / 2 for x0, x1, y0, y1 in pairs)


async def deletion_curve(
    config: ExplainConfig,
    snapshot: ConversationSnapshot,
    attributions: Sequence[float],
    ks: Sequence[int] | None = None,
    registry: PluginRegistry | None = None,
    backend: GenerativeBackend | None = None,
) -> DeletionCurve:
    """Score ``snapshot`` with its ``k`` highest-attributed players removed at once.

    Players are ranked by ``attributions`` (ties keep player order) and each
    ``k`` in ``ks`` (default ``0..n``) removes the top ``k`` together, never
    greedily, using ``config``'s partition, absence policy, backend and value
    function. Use a different value function than the explanation's to avoid a
    circular check. A ``backend`` (for example the explained run's loaded model)
    is reused and left open; without one the run builds and closes its own.

    Raises:
        ValueError: If ``attributions`` does not have one value per player, or a
            ``k`` is outside ``0..n``.
    """
    sampler = _DeletionSampler(attributions, ks)
    plugins = (registry or default_registry()).copy()
    plugins.register(PluginGroup.ESTIMATORS, _ESTIMATOR, lambda _context: sampler)
    explanation = config.explanation.model_copy(
        update={"estimator": _ESTIMATOR, "budget": BudgetConfig(evaluations=1)}
    )
    run_config = config.model_copy(update={"explanation": explanation})
    runner = ExplainRunner(run_config, registry=plugins, backend=backend)
    output = await runner.explain(snapshot)
    return DeletionCurve(
        num_players=len(attributions),
        grand_value=output.result.grand_value,
        removed=sampler.ks,
        values=tuple(sampler.payoffs[mask] for mask in sampler.masks),
    )


class _DeletionSampler(_SnapshotEstimator):
    """Estimator that evaluates exactly the deletion coalitions and keeps payoffs."""

    def __init__(self, attributions: Sequence[float], ks: Sequence[int] | None) -> None:
        super().__init__()
        self._attributions = list(attributions)
        self._ks = ks
        self.ks: tuple[int, ...] = ()
        self.masks: list[tuple[bool, ...]] = []
        self.payoffs: dict[tuple[bool, ...], float] = {}

    @property
    def name(self) -> str:
        return _ESTIMATOR

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        _ = budget_fraction, include_minimal_masks, seed
        n = player_set.num_players
        if len(self._attributions) != n:
            msg = (
                f"need one attribution per player ({n}), got {len(self._attributions)}"
            )
            raise ValueError(msg)
        self.ks = tuple(sorted(range(n + 1) if self._ks is None else self._ks))
        if any(k < 0 or k > n for k in self.ks):
            msg = f"every k must lie in 0 to {n}, got {self.ks}"
            raise ValueError(msg)
        order = sorted(range(n), key=lambda i: -self._attributions[i])
        for k in self.ks:
            removed = set(order[:k])
            present = tuple(i not in removed for i in range(n))
            self.masks.append(present)
            yield CoalitionMask.from_sequence(present)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        _ = aggregator
        self.payoffs = {m.present: v for m, v in zip(masks, payoffs, strict=True)}
        return [0.0] * len(self._attributions)
