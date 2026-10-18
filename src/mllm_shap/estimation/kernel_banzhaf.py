"""Kernel Banzhaf: paired uniform sampling with a centred linear regression.

The Banzhaf value of a player is the slope of the best linear fit of ``v`` on the
presence indicators under the uniform distribution over coalitions (Wang and Jia
2023). The design uses indicators shifted by one half and an intercept, so the
columns are orthogonal on the full table and the fit is exact there.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_kernel_banzhaf.py]]
"""

from collections.abc import Iterator, Sequence

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.estimands import Estimand
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from ._shared import (
    _SnapshotEstimator,
    compute_complementary_num_splits,
    is_empty_or_grand,
    present_to_mask_int,
)


class KernelBanzhafEstimator(_SnapshotEstimator):
    """Sample uniform complementary pairs and fit the Banzhaf regression."""

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "kernel_banzhaf"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield ``(N, empty)`` then distinct uniform pairs ``(S, N - S)``."""
        num_players = player_set.num_players
        total = compute_complementary_num_splits(
            num_players, budget_fraction, include_minimal_masks
        )
        rng = np.random.default_rng(seed)
        grand = (True,) * num_players
        empty = (False,) * num_players
        seen = {present_to_mask_int(grand), present_to_mask_int(empty)}
        generated = 2
        yield CoalitionMask.from_sequence(grand)
        yield CoalitionMask.from_sequence(empty)
        while generated + 2 <= total:
            present = tuple(bool(bit) for bit in rng.integers(0, 2, num_players))
            complement = tuple(not value for value in present)
            if is_empty_or_grand(present) or present_to_mask_int(present) in seen:
                continue
            seen.add(present_to_mask_int(present))
            seen.add(present_to_mask_int(complement))
            generated += 2
            yield CoalitionMask.from_sequence(present)
            yield CoalitionMask.from_sequence(complement)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Return the slopes of the least-squares fit of the payoffs on the rows.

        Raises:
            ValueError: If the estimand is not Banzhaf.
        """
        if aggregator.estimand is not Estimand.BANZHAF:
            msg = (
                f"{self.name} supports the banzhaf estimand only, "
                f"got {aggregator.estimand}"
            )
            raise ValueError(msg)
        rows = np.array([mask.present for mask in masks], dtype=np.float64) - 0.5
        design = np.column_stack([np.ones(len(rows)), rows])
        target = np.asarray(payoffs, dtype=np.float64)
        solution = np.linalg.lstsq(design, target, rcond=None)[0]
        return [float(value) for value in solution[1:]]
