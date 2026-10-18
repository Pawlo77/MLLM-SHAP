"""Monte Carlo coalition sampling estimator.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_monte_carlo.py]]
"""

import math
from collections.abc import Iterator
from itertools import chain

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ._shared import (
    _SnapshotEstimator,
    compute_mc_num_samples,
    iter_singletons,
    present_to_mask_int,
    random_present,
)


class MonteCarloEstimator(_SnapshotEstimator):
    """Sample random coalitions and delegate attribution to an estimand plugin."""

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "mc"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield the small strata exactly, then coalitions with uniformly drawn sizes.

        Order (Stratified SVARM): the grand and empty coalitions, the singletons,
        the coalitions missing one player, then a size drawn uniformly from the
        sizes with unseen coalitions and a uniform coalition of that size, without
        repeats. All count toward the budget.
        """
        num_players = player_set.num_players
        num_samples = compute_mc_num_samples(
            num_players,
            budget_fraction,
            include_minimal_masks,
        )
        rng = np.random.default_rng(seed)
        seen: set[int] = set()
        seen_by_size = [0] * (num_players + 1)
        generated = 0

        def take(present: tuple[bool, ...]) -> bool:
            nonlocal generated
            mask_int = present_to_mask_int(present)
            if mask_int in seen or generated >= num_samples:
                return False
            seen.add(mask_int)
            seen_by_size[sum(present)] += 1
            generated += 1
            return True

        # v(N) and v(empty) are needed by every estimand; the exact strata of
        # sizes 1 and n - 1 follow because their means cost only n evaluations each.
        exact = chain(
            ((True,) * num_players, (False,) * num_players),
            iter_singletons(num_players),
            (
                tuple(not value for value in present)
                for present in iter_singletons(num_players)
            ),
        )
        for present in exact:
            if take(present):
                yield CoalitionMask.from_sequence(present)

        open_sizes = [
            size
            for size in range(2, num_players - 1)
            if seen_by_size[size] < math.comb(num_players, size)
        ]
        while generated < num_samples and open_sizes:
            size = open_sizes[int(rng.integers(len(open_sizes)))]
            present = random_present(rng, num_players, true_count=size)
            if take(present):
                yield CoalitionMask.from_sequence(present)
            if seen_by_size[size] == math.comb(num_players, size):
                open_sizes.remove(size)
