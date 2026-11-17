"""Removal metrics: how fast a ranking of players destroys or restores the answer.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_faithfulness.py]]
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..estimation._shared import _SnapshotEstimator
from ..plugins import PluginGroup, PluginRegistry, default_registry
from ..protocols.backend import GenerativeBackend
from ..protocols.estimand import EstimandAggregator
from .config import BudgetConfig, ExplainConfig
from .runner import ExplainRunner

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class RemovalMetrics:
    """Removal metrics of one ranking, read off one set of scored coalitions.

    ``deletion_values[k]`` is ``v`` with the top-``k`` players absent and
    ``insertion_values[k]`` is ``v`` with only the top-``k`` present.
    """

    num_players: int
    """Number of players of the explained snapshot."""

    grand_value: float
    """``v(N)``."""

    empty_value: float
    """``v(empty)``."""

    ks: tuple[int, ...]
    """Numbers of top-ranked players, ascending."""

    deletion_values: tuple[float, ...]
    """``v(N - top_k)`` for each ``k`` in :attr:`ks`."""

    insertion_values: tuple[float, ...]
    """``v(top_k)`` for each ``k`` in :attr:`ks`."""

    deletion_win_rate: float | None
    """Share of size-matched random sets whose removal drops ``v`` less than the
    ranking's top-``k`` does (ties count half), averaged over ``k``; ``None``
    without random sets."""

    insertion_win_rate: float | None
    """Share of size-matched random sets that restore ``v`` less than the ranking's
    top-``k`` does (ties count half), averaged over ``k``."""

    flip_size: int | None
    """Smallest ``k`` whose removal takes ``v`` below the threshold; ``None`` when
    no threshold was given, ``v(N)`` is already below it, or no ``k`` flips it."""

    restore_size: int | None
    """Smallest ``k`` whose insertion alone lifts ``v`` to the threshold."""

    explained_with: tuple[str, str] | None
    """Value function and absence policy of the explanation, when given."""

    evaluated_with: tuple[str, str]
    """Value function and absence policy these metrics were scored with."""

    @property
    def circular(self) -> bool:
        """Return whether the metric reuses the explanation's game and filler."""
        return self.explained_with == self.evaluated_with

    def comprehensiveness(self, k: int) -> float:
        """Return ``v(N) - v(N - top_k)`` (ERASER); higher is more faithful."""
        return self.grand_value - self.deletion_values[self.ks.index(k)]

    def sufficiency(self, k: int) -> float:
        """Return ``v(N) - v(top_k)`` (ERASER); lower is more faithful."""
        return self.grand_value - self.insertion_values[self.ks.index(k)]

    @property
    def aopc(self) -> float:
        """Return the mean comprehensiveness over ``k >= 1`` over ``v(N) - v(empty)``.

        The scale maps full removal to 1 for a game whose value falls to ``v(empty)``;
        zero when the grand and empty values coincide.
        """
        span = self.grand_value - self.empty_value
        drops = [self.comprehensiveness(k) for k in self.ks if k >= 1]
        if span == 0.0 or not drops:
            return 0.0
        return sum(drops) / len(drops) / span

    @property
    def deletion_area(self) -> float:
        """Return the area of the deletion curve over ``k / n``; lower is better."""
        return _area(self.ks, self.deletion_values, self.num_players)

    @property
    def insertion_area(self) -> float:
        """Return the area of the insertion curve over ``k / n``; higher is better."""
        return _area(self.ks, self.insertion_values, self.num_players)


def _area(ks: Sequence[int], values: Sequence[float], num_players: int) -> float:
    xs = [k / num_players for k in ks]
    pairs = zip(xs, xs[1:], values, values[1:], strict=False)
    return sum((x1 - x0) * (y0 + y1) / 2 for x0, x1, y0, y1 in pairs)


def _random_sets(
    n: int, ks: Sequence[int], count: int, seed: int
) -> dict[int, list[set[int]]]:
    """Return ``count`` seeded random size-``k`` player sets for each ``k``."""
    rng = np.random.default_rng(seed)
    return {
        k: [
            {int(i) for i in rng.choice(n, size=k, replace=False)} for _ in range(count)
        ]
        for k in ks
    }


def removal_plan(
    attributions: Sequence[float],
    ks: Sequence[int] | None = None,
    random_sets: int = 0,
    seed: int = 0,
) -> list[tuple[bool, ...]]:
    """Return every coalition the removal metrics read, without duplicates.

    Per ``k``: the top-``k`` removed, the top-``k`` kept alone, and for each of
    ``random_sets`` seeded random size-``k`` sets the same two coalitions. The grand
    and empty coalitions are always included.

    Raises:
        ValueError: If a ``k`` lies outside ``0..n``.
    """
    n = len(attributions)
    chosen = tuple(sorted(range(n + 1) if ks is None else ks))
    if any(k < 0 or k > n for k in chosen):
        msg = f"every k must lie in 0 to {n}, got {chosen}"
        raise ValueError(msg)
    order = sorted(range(n), key=lambda i: -attributions[i])
    random = _random_sets(n, chosen, random_sets, seed)
    sets: list[set[int]] = []
    for k in chosen:
        sets.append(set(order[:k]))
        sets.extend(random[k])
    coalitions = [(True,) * n, (False,) * n]
    for chosen_set in sets:
        coalitions.append(tuple(i not in chosen_set for i in range(n)))
        coalitions.append(tuple(i in chosen_set for i in range(n)))
    return list(dict.fromkeys(coalitions))


def compute_removal_metrics(
    attributions: Sequence[float],
    payoff: Callable[[tuple[bool, ...]], float],
    ks: Sequence[int] | None = None,
    random_sets: int = 0,
    seed: int = 0,
    threshold: float | None = None,
    explained_with: tuple[str, str] | None = None,
    evaluated_with: tuple[str, str] = ("", ""),
) -> RemovalMetrics:
    """Return the removal metrics of a ranking from a coalition payoff function.

    ``payoff`` must answer every coalition of :func:`removal_plan` with the same
    arguments; ``threshold`` is the decision boundary of a closed-answer game, such as
    0 for ``label_margin``. A warning is logged when the evaluation reuses the
    explanation's value function and absence policy.
    """
    n = len(attributions)
    chosen = tuple(sorted(range(n + 1) if ks is None else ks))
    plan = removal_plan(attributions, chosen, random_sets, seed)
    values = {mask: payoff(mask) for mask in plan}
    order = sorted(range(n), key=lambda i: -attributions[i])

    def removed(members: set[int]) -> float:
        return values[tuple(i not in members for i in range(n))]

    def kept(members: set[int]) -> float:
        return values[tuple(i in members for i in range(n))]

    grand, empty = values[(True,) * n], values[(False,) * n]
    deletion = tuple(removed(set(order[:k])) for k in chosen)
    insertion = tuple(kept(set(order[:k])) for k in chosen)

    random = _random_sets(n, chosen, random_sets, seed)

    def win_rate(
        mine: Sequence[float], read: Callable[[set[int]], float], lower: bool
    ) -> float | None:
        wins = trials = 0.0
        for k, own in zip(chosen, mine, strict=True):
            if k in (0, n):
                continue
            for members in random[k]:
                other = read(members)
                better = own < other if lower else own > other
                wins += 1.0 if better else (0.5 if own == other else 0.0)
                trials += 1.0
        return wins / trials if trials else None

    flip = restore = None
    if threshold is not None:
        if grand >= threshold:
            flip = next(
                (k for k, v in zip(chosen, deletion, strict=True) if v < threshold),
                None,
            )
        restore = next(
            (k for k, v in zip(chosen, insertion, strict=True) if v >= threshold), None
        )
    metrics = RemovalMetrics(
        num_players=n,
        grand_value=grand,
        empty_value=empty,
        ks=chosen,
        deletion_values=deletion,
        insertion_values=insertion,
        deletion_win_rate=win_rate(deletion, removed, lower=True),
        insertion_win_rate=win_rate(insertion, kept, lower=False),
        flip_size=flip,
        restore_size=restore,
        explained_with=explained_with,
        evaluated_with=evaluated_with,
    )
    if metrics.circular:
        logger.warning(
            "removal metrics are scored with the explanation's value function and "
            "absence policy %s; they favour the explainer by construction",
            evaluated_with,
        )
    return metrics


class PlanSampler(_SnapshotEstimator):
    """Estimator that evaluates a fixed list of coalitions and keeps their payoffs."""

    def __init__(self, build: Callable[[int], list[tuple[bool, ...]]]) -> None:
        """Plan the coalitions with ``build(num_players)`` when sampling starts."""
        super().__init__()
        self._build = build
        self.payoffs: dict[tuple[bool, ...], float] = {}

    @property
    def name(self) -> str:
        """Return the private estimator identifier."""
        return "_removal_metrics"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield exactly the planned coalitions, ignoring the budget."""
        _ = budget_fraction, include_minimal_masks, seed
        for present in self._build(player_set.num_players):
            yield CoalitionMask.from_sequence(present)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Keep the payoffs by coalition; the attributions are placeholders."""
        _ = aggregator
        self.payoffs = {m.present: v for m, v in zip(masks, payoffs, strict=True)}
        return [0.0] * len(masks[0].present)


async def removal_metrics(
    config: ExplainConfig,
    snapshot: ConversationSnapshot,
    attributions: Sequence[float],
    ks: Sequence[int] | None = None,
    random_sets: int = 20,
    threshold: float | None = None,
    explained_with: ExplainConfig | None = None,
    registry: PluginRegistry | None = None,
    backend: GenerativeBackend | None = None,
) -> RemovalMetrics:
    """Score the removal metrics of ``attributions`` under ``config``.

    One run evaluates every coalition of :func:`removal_plan` (deduplicated, and
    answered from an archive when ``config`` points to one). Pass the explanation's
    config as ``explained_with`` so a metric that reuses its value function and
    absence policy is flagged.
    """
    seed = config.explanation.seed
    sampler = PlanSampler(lambda _n: removal_plan(attributions, ks, random_sets, seed))
    plugins = (registry or default_registry()).copy()
    plugins.register(PluginGroup.ESTIMATORS, sampler.name, lambda _context: sampler)
    explanation = config.explanation.model_copy(
        update={"estimator": sampler.name, "budget": BudgetConfig(evaluations=1)}
    )
    run_config = config.model_copy(update={"explanation": explanation})
    await ExplainRunner(run_config, registry=plugins, backend=backend).explain(snapshot)
    return compute_removal_metrics(
        attributions,
        sampler.payoffs.__getitem__,
        ks,
        random_sets,
        seed,
        threshold,
        None
        if explained_with is None
        else (
            explained_with.explanation.value_fn,
            explained_with.explanation.absence_policy,
        ),
        (config.explanation.value_fn, config.explanation.absence_policy),
    )
