"""Leverage SHAP: paired, leverage-score sampling with a constrained regression.

Every coalition size carries the same total leverage in the Kernel SHAP
regression (Musco and Witter 2025), so sizes are sampled uniformly and each
size stratum is weighted by its population over its sample count. Pairs
``(S, N - S)`` are drawn without replacement; a stratum whose share of the
budget covers all its pairs is enumerated.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_leverage.py]]
"""

import itertools
import math
from collections.abc import Iterator, Sequence

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.estimands import Estimand
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from ._shared import (
    _SnapshotEstimator,
    compute_complementary_num_splits,
    present_to_mask_int,
)


def pair_classes(num_players: int) -> list[tuple[int, int]]:
    """Return ``(size, pairs)`` per pair class of sizes ``s`` and ``n - s``."""
    return [
        (size, math.comb(num_players, size) // (2 if 2 * size == num_players else 1))
        for size in range(1, num_players // 2 + 1)
    ]


def allocate_pairs(num_players: int, pairs: int) -> dict[int, int]:
    """Split ``pairs`` over the pair classes in proportion to their leverage mass.

    A class holding two sizes carries twice the mass of the middle class. Classes
    whose share reaches their population are enumerated and drop out; the
    rest get whole counts by largest remainder.
    """
    classes = dict(pair_classes(num_players))
    units = {size: 1 if 2 * size == num_players else 2 for size in classes}
    allocation: dict[int, int] = {}
    left = pairs
    while True:
        total = sum(units[size] for size in classes if size not in allocation)
        full = [
            size
            for size in classes
            if size not in allocation and left * units[size] >= classes[size] * total
        ]
        if not full:
            break
        for size in full:
            allocation[size] = classes[size]
            left -= classes[size]
    rest = [size for size in classes if size not in allocation]
    total = sum(units[size] for size in rest)
    shares = {size: left * units[size] / total for size in rest} if rest else {}
    for size in rest:
        allocation[size] = int(shares[size])
    leftover = left - sum(allocation[size] for size in rest)
    for size in sorted(rest, key=lambda s: shares[s] - allocation[s], reverse=True):
        if leftover <= 0:
            break
        if allocation[size] < classes[size]:
            allocation[size] += 1
            leftover -= 1
    return allocation


def distinct_subsets(
    rng: np.random.Generator,
    num_players: int,
    size: int,
    count: int,
    contains_first: bool,
) -> Iterator[tuple[bool, ...]]:
    """Yield ``count`` distinct uniform subsets of ``size`` players.

    ``contains_first`` restricts to subsets holding player 0, one per pair when
    both sides have the same size.
    """
    others = range(1, num_players) if contains_first else range(num_players)
    pick = size - 1 if contains_first else size
    population = math.comb(len(others), pick)
    if population <= 4 * count or population <= 4096:
        chosen = rng.choice(population, size=count, replace=False)
        wanted = {int(index) for index in chosen}
        for index, combo in enumerate(itertools.combinations(others, pick)):
            if index in wanted:
                present = [False] * num_players
                present[0] = contains_first
                for player in combo:
                    present[player] = True
                yield tuple(present)
        return
    seen: set[int] = set()
    while len(seen) < count:
        chosen = rng.choice(len(others), size=pick, replace=False)
        present = [False] * num_players
        present[0] = contains_first
        for index in chosen:
            present[others[int(index)]] = True
        key = present_to_mask_int(present)
        if key not in seen:
            seen.add(key)
            yield tuple(present)


class LeverageEstimator(_SnapshotEstimator):
    """Sample complementary pairs by leverage and solve the Kernel SHAP regression."""

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "leverage"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield ``(N, empty)`` then complementary pairs per size stratum."""
        num_players = player_set.num_players
        total = compute_complementary_num_splits(
            num_players, budget_fraction, include_minimal_masks
        )
        grand = (True,) * num_players
        yield CoalitionMask.from_sequence(grand)
        yield CoalitionMask.from_sequence((False,) * num_players)
        rng = np.random.default_rng(seed)
        allocation = allocate_pairs(num_players, (total - 2) // 2)
        for size, count in allocation.items():
            for present in distinct_subsets(
                rng, num_players, size, count, contains_first=2 * size == num_players
            ):
                yield CoalitionMask.from_sequence(present)
                yield CoalitionMask.from_sequence(tuple(not v for v in present))

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Solve the efficiency-constrained, stratum-weighted least squares.

        Raises:
            ValueError: If the estimand is not Shapley or a boundary mask is missing.
        """
        if aggregator.estimand is not Estimand.SHAPLEY:
            msg = (
                f"{self.name} supports the shapley estimand only, "
                f"got {aggregator.estimand}"
            )
            raise ValueError(msg)
        return leverage_values(
            np.array([mask.present for mask in masks], dtype=np.float64), payoffs
        )


def leverage_values(present: np.ndarray, payoffs: Sequence[float]) -> list[float]:
    """Return Shapley estimates from coalition rows by constrained weighted regression.

    Stratum ``s`` weighs ``(n - 1) / (s (n - s) count_s)``: the Shapley kernel
    times the stratum population over its sample count, which is exact when the
    stratum is enumerated. The sum constraint is solved by elimination of the
    last player.
    """
    num_players = present.shape[1]
    pay = np.asarray(payoffs, dtype=np.float64)
    sizes = present.sum(axis=1).astype(np.int64)
    grand = np.flatnonzero(sizes == num_players)
    empty = np.flatnonzero(sizes == 0)
    if grand.size == 0 or empty.size == 0:
        msg = "masks must include the grand and the empty coalition"
        raise ValueError(msg)
    delta = pay[grand[0]] - pay[empty[0]]
    if num_players == 1:
        return [float(delta)]
    inner = (sizes > 0) & (sizes < num_players)
    rows, inner_sizes = present[inner], sizes[inner]
    counts = np.bincount(inner_sizes, minlength=num_players + 1)
    spread = inner_sizes * (num_players - inner_sizes) * counts[inner_sizes]
    weights = (num_players - 1) / spread
    design = rows[:, :-1] - rows[:, [-1]]
    target = pay[inner] - pay[empty[0]] - rows[:, -1] * delta
    scale = np.sqrt(weights)
    solution = np.linalg.lstsq(design * scale[:, None], target * scale, rcond=None)[0]
    return [*solution.tolist(), float(delta - solution.sum())]
