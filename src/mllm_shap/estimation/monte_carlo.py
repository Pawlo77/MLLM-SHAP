"""Monte Carlo coalition sampling estimator.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_monte_carlo.py]]
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ._shared import (
    _SnapshotEstimator,
    compute_mc_num_samples,
    iter_minimal_masks,
    present_to_mask_int,
    random_present,
)

if TYPE_CHECKING:
    pass


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
        """Yield the grand and empty coalitions, then random ones, up to the budget."""
        num_players = player_set.num_players
        num_samples = compute_mc_num_samples(
            num_players,
            budget_fraction,
            include_minimal_masks,
        )
        rng = np.random.default_rng(seed)
        seen: set[int] = set()
        generated = 0

        # v(N) and v(empty) are needed by every estimand; they come first and
        # count toward the budget.
        for present in ((True,) * num_players, (False,) * num_players):
            seen.add(present_to_mask_int(present))
            generated += 1
            yield CoalitionMask.from_sequence(present)

        if include_minimal_masks:
            for present in iter_minimal_masks(num_players):
                mask_int = present_to_mask_int(present)
                if mask_int in seen:
                    continue
                seen.add(mask_int)
                generated += 1
                yield CoalitionMask.from_sequence(present)
                if generated >= num_samples:
                    return

        while generated < num_samples:
            if len(seen) >= (1 << num_players):
                return
            present = random_present(rng, num_players)
            mask_int = present_to_mask_int(present)
            if mask_int in seen:
                continue
            seen.add(mask_int)
            generated += 1
            yield CoalitionMask.from_sequence(present)
