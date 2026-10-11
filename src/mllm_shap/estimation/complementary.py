"""Complementary pair sampling estimator.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_complementary.py]]
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.estimands import Estimand
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from ._shared import (
    _SnapshotEstimator,
    aggregate_complementary_shapley,
    build_c_matrix_from_pairs,
    build_m_counts_from_masks,
    complementary_base_present,
    compute_complementary_num_splits,
    increment_m_counts,
    is_empty_or_grand,
    iter_singletons,
    present_to_mask_int,
    random_present,
)

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class ComplementaryEstimator(_SnapshotEstimator):
    """Sample complementary coalition pairs and aggregate CC attributions."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        super().__init__(context)
        self._m_counts: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "complementary"

    @property
    def m_counts(self) -> np.ndarray | None:
        """Return the latest complementary M-matrix counts from sampling."""
        return None if self._m_counts is None else self._m_counts.copy()

    def reset_sampling_state(self, num_players: int) -> None:
        """Reset complementary M-matrix counts before a new sampling run."""
        self._m_counts = np.zeros((num_players, num_players + 1), dtype=np.int64)

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield complementary coalition pairs up to the configured budget.

        Order: the ``(N, empty)`` pair, the singleton pairs ``({i}, N - {i})``
        (the ``2n + 2`` budget floor fits all of them), then random pairs. All
        count toward the budget.
        """
        num_players = player_set.num_players
        num_splits = compute_complementary_num_splits(
            num_players,
            budget_fraction,
            include_minimal_masks,
        )
        self.reset_sampling_state(num_players)
        m_counts = self._m_counts
        if m_counts is None:
            msg = "M matrix must be initialized before sampling."
            raise RuntimeError(msg)
        rng = np.random.default_rng(seed)
        seen: set[int] = set()
        generated = 0
        base_index = 0

        grand = (True,) * num_players
        empty = (False,) * num_players
        seen.add(present_to_mask_int(grand))
        seen.add(present_to_mask_int(empty))
        increment_m_counts(m_counts, grand)
        increment_m_counts(m_counts, empty)
        generated += 2
        yield CoalitionMask.from_sequence(grand)
        yield CoalitionMask.from_sequence(empty)

        # Only the ({i}, N - {i}) pair fills player i's size-1 stratum; seed
        # them all before random pairs so no stratum is left empty (zero-filled).
        for present in iter_singletons(num_players):
            if generated + 2 > num_splits:
                return
            complement = tuple(not value for value in present)
            mask_int = present_to_mask_int(present)
            complement_int = present_to_mask_int(complement)
            if is_empty_or_grand(present) or mask_int in seen or complement_int in seen:
                continue
            seen.add(mask_int)
            seen.add(complement_int)
            increment_m_counts(m_counts, present)
            increment_m_counts(m_counts, complement)
            generated += 2
            yield CoalitionMask.from_sequence(present)
            yield CoalitionMask.from_sequence(complement)

        while generated < num_splits:
            if include_minimal_masks and base_index < num_players:
                present = complementary_base_present(rng, num_players, base_index)
                base_index += 1
            else:
                present = random_present(rng, num_players)

            if is_empty_or_grand(present):
                continue

            mask_int = present_to_mask_int(present)
            complement_int = present_to_mask_int(tuple(not value for value in present))
            if mask_int in seen or complement_int in seen:
                continue

            if generated + 2 > num_splits:
                return

            complement = tuple(not value for value in present)
            seen.add(mask_int)
            seen.add(complement_int)

            increment_m_counts(m_counts, present)
            increment_m_counts(m_counts, complement)
            generated += 2
            yield CoalitionMask.from_sequence(present)
            yield CoalitionMask.from_sequence(complement)

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Aggregate complementary pairs into Shapley attributions.

        Masks are paired by complement, so their order does not matter.
        Complementary contributions estimate the Shapley value only.

        Raises:
            ValueError: If the estimand is not Shapley, ``masks`` is empty, or a
                mask has no complement in ``masks``.
        """
        if aggregator.estimand is not Estimand.SHAPLEY:
            msg = (
                f"{self.name} supports the shapley estimand only, "
                f"got {aggregator.estimand}"
            )
            raise ValueError(msg)
        if not masks:
            msg = "masks must not be empty"
            raise ValueError(msg)
        num_players = len(masks[0].present)
        m_counts = build_m_counts_from_masks(masks, num_players)
        c_matrix = build_c_matrix_from_pairs(m_counts, masks, payoffs)
        return aggregate_complementary_shapley(m_counts, c_matrix)
