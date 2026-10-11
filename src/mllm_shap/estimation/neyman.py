"""Neyman complementary allocation estimator.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_neyman.py]]
"""

import math
from collections.abc import Iterator, Sequence
from enum import IntEnum
from typing import TYPE_CHECKING, cast

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ._shared import (
    build_c_matrix_from_pairs,
    build_c_squared_from_pairs,
    compute_complementary_num_splits,
    increment_m_counts,
    iter_singletons,
    present_to_mask_int,
    random_present,
)
from .complementary import ComplementaryEstimator

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class _NeymanStep(IntEnum):
    """Phases in the Neyman allocation process."""

    INITIAL_SAMPLING = 1
    NEYMAN_ALLOCATION = 2


class NeymanEstimator(ComplementaryEstimator):
    """Two-phase complementary estimator with Neyman coalition-size allocation."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        initial_fraction: float | None = None,
        initial_num_samples: int | None = None,
        use_standard_method: bool = False,
    ) -> None:
        """Configure phase-one sizing.

        With a ``context`` the settings come from ``explanation.neyman`` and the
        keyword arguments are ignored.
        """
        super().__init__(context)
        if context is not None:
            neyman = context.config.explanation.neyman
            initial_fraction = neyman.initial_fraction
            initial_num_samples = neyman.initial_num_samples
            use_standard_method = neyman.use_standard_method
        self.initial_fraction = initial_fraction
        self.initial_num_samples = initial_num_samples
        self.use_standard_method = use_standard_method
        self._use_default_initial_formula = (
            initial_fraction is None and initial_num_samples is None
        )
        self._step = _NeymanStep.INITIAL_SAMPLING
        self._initial_num_splits = 0
        self._position_i = 0
        self._position_j = 0
        self._m_hat: np.ndarray | None = None
        self._c_matrix: np.ndarray | None = None
        self._c_squared: np.ndarray | None = None
        self._total_num_splits = 0
        self._num_players = 0
        self._rng: np.random.Generator | None = None
        self._seen: set[int] | None = None

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "neyman_cc"

    def _resolve_initial_num_splits(self, num_players: int, total_splits: int) -> int:
        if self._use_default_initial_formula:
            initial = max(2, math.ceil(total_splits / (2 * num_players * num_players)))
        elif self.initial_num_samples is not None:
            initial = self.initial_num_samples
        elif self.initial_fraction is not None:
            initial = int(
                compute_complementary_num_splits(
                    num_players,
                    self.initial_fraction,
                    include_minimal_masks=False,
                )
            )
        else:
            msg = "initial_fraction or initial_num_samples must be set."
            raise ValueError(msg)
        return max(2, initial)

    def _reset_neyman_state(
        self,
        num_players: int,
        budget_fraction: float,
        seed: int,
    ) -> None:
        self.reset_sampling_state(num_players)
        self._num_players = num_players
        self._total_num_splits = compute_complementary_num_splits(
            num_players,
            budget_fraction,
            include_minimal_masks=False,
        )
        self._initial_num_splits = self._resolve_initial_num_splits(
            num_players,
            self._total_num_splits,
        )
        self._step = _NeymanStep.INITIAL_SAMPLING
        self._position_i = 0
        # Sizes 0 and n are empty/grand and are never yielded; start at size 1.
        self._position_j = 1 if num_players >= 2 else 0
        self._m_hat = None
        self._c_matrix = None
        self._c_squared = None
        self._rng = np.random.default_rng(seed)
        self._seen = set()

    def _allocation_start_index(self) -> int:
        m_counts = self._m_counts
        if m_counts is None:
            msg = "M matrix must be initialized before allocation."
            raise RuntimeError(msg)
        return cast(int, math.ceil(m_counts.shape[0] / 2))

    @staticmethod
    def _complement_column(coalition_size: int, num_players: int) -> int:
        """Return the M-matrix column holding the complement of a coalition."""
        return num_players - coalition_size

    def _cell_targets(self) -> np.ndarray:
        """Per-cell phase-one targets capped by the number of distinct coalitions."""
        n = self._num_players
        targets = np.full((n, n + 1), self._initial_num_splits, dtype=np.int64)
        for size in range(n + 1):
            # Distinct size-``size`` coalitions containing a fixed player.
            feasible = math.comb(n - 1, size - 1) if size >= 1 else 1
            targets[:, size] = np.minimum(targets[:, size], feasible)
        return targets

    def _update_m_position(self) -> bool:
        """Advance the initial-phase cursor.

        Return True when initial sampling completes.

        Cells are visited size by size and, within a size, player by player.
        Empty (size 0) and grand (size n) columns are ignored: the ``(N, empty)``
        pair is sampled once up front. Every other cell targets
        ``initial_num_splits`` capped by the number of distinct coalitions that
        can fill it, so phase one always terminates.
        """
        m_counts = self._m_counts
        if m_counts is None:
            msg = "M matrix must be initialized before sampling."
            raise RuntimeError(msg)
        incomplete = m_counts < self._cell_targets()
        incomplete[:, 0] = False
        incomplete[:, -1] = False
        if not np.any(incomplete):
            return True

        # Walk the grid size by size, round-robin over players, so a budget that
        # truncates phase one leaves the same strata empty for every player.
        num_players = m_counts.shape[0]
        flat = incomplete.T.reshape(-1)
        start_index = self._position_j * num_players + self._position_i + 1
        next_indices = np.flatnonzero(flat[start_index:])
        if next_indices.size == 0:
            next_indices = np.flatnonzero(flat)
            start_index = 0
        next_index = int(next_indices[0]) + start_index
        self._position_j, self._position_i = divmod(next_index, num_players)
        return False

    def _next_initial_split(self) -> tuple[bool, ...]:
        rng = self._rng
        m_counts = self._m_counts
        if rng is None or m_counts is None:
            msg = "Neyman sampling state must be initialized."
            raise RuntimeError(msg)
        coalition_size = self._position_j
        player_index = self._position_i

        if self.use_standard_method:
            return random_present(
                rng,
                self._num_players,
                true_count=coalition_size,
            )

        include_index = player_index if coalition_size > 0 else None
        present = random_present(
            rng,
            self._num_players,
            true_count=coalition_size,
            include_index=include_index,
        )
        if coalition_size > 0 and not present[player_index]:
            msg = "Generated mask does not include the required player."
            raise RuntimeError(msg)
        return present

    def _next_allocation_split(self) -> tuple[bool, ...] | None:
        rng = self._rng
        m_hat = self._m_hat
        if rng is None or m_hat is None:
            msg = "Neyman allocation state must be initialized."
            raise RuntimeError(msg)
        while self._position_j < m_hat.shape[0]:
            if m_hat[self._position_j] > 0:
                return random_present(
                    rng,
                    self._num_players,
                    true_count=self._position_j,
                )
            self._position_j += 1
        return None

    def _unseen_pairs_by_size(self) -> np.ndarray:
        """Count complementary pairs not yet sampled, indexed by larger side size."""
        seen = self._seen
        if seen is None:
            msg = "Neyman sampling state must be initialized."
            raise RuntimeError(msg)
        n = self._num_players
        seen_by_size = np.zeros(n + 1, dtype=np.int64)
        for mask_int in seen:
            seen_by_size[mask_int.bit_count()] += 1
        unseen = np.zeros(n + 1, dtype=np.int64)
        for size in range(self._allocation_start_index(), n):
            total = math.comb(n, size)
            if 2 * size == n:
                # Both sides have size n/2: each pair holds two such coalitions.
                unseen[size] = (total - seen_by_size[size]) // 2
            else:
                unseen[size] = total - seen_by_size[size]
        return unseen

    def _yield_complementary_pair(
        self,
        present: tuple[bool, ...],
    ) -> Iterator[CoalitionMask]:
        m_counts = self._m_counts
        seen = self._seen
        if m_counts is None or seen is None:
            msg = "Neyman sampling state must be initialized."
            raise RuntimeError(msg)
        complement = tuple(not value for value in present)
        mask_int = present_to_mask_int(present)
        complement_int = present_to_mask_int(complement)
        if mask_int in seen or complement_int in seen:
            return
        seen.add(mask_int)
        seen.add(complement_int)
        increment_m_counts(m_counts, present)
        increment_m_counts(m_counts, complement)
        yield CoalitionMask.from_sequence(present)
        yield CoalitionMask.from_sequence(complement)

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield phase-one Neyman masks until the initial M-matrix grid is filled.

        Order: the ``(N, empty)`` pair, the singleton pairs ``({i}, N - {i})``,
        then the grid walk over (size, player) cells. All count toward the budget.
        """
        if include_minimal_masks:
            msg = "Neyman estimator does not support include_minimal_masks=True"
            raise ValueError(msg)
        self._reset_neyman_state(player_set.num_players, budget_fraction, seed)
        m_counts = self._m_counts
        seen = self._seen
        if m_counts is None or seen is None:
            msg = "Neyman sampling state must be initialized."
            raise RuntimeError(msg)
        num_players = player_set.num_players
        grand = (True,) * num_players
        empty = (False,) * num_players
        seen.add(present_to_mask_int(grand))
        seen.add(present_to_mask_int(empty))
        increment_m_counts(m_counts, grand)
        increment_m_counts(m_counts, empty)
        generated = 2
        yield CoalitionMask.from_sequence(grand)
        yield CoalitionMask.from_sequence(empty)
        # The singleton pairs are the only way to fill each size-1 stratum; seed
        # them before the grid walk; the ``2n + 2`` budget floor fits all of them.
        for singleton in iter_singletons(num_players):
            if generated + 2 > self._total_num_splits:
                break
            for mask in self._yield_complementary_pair(singleton):
                generated += 1
                yield mask
        stale_attempts = 0
        # One unproductive visit per grid cell in a row ends phase one.
        max_stale = m_counts.size
        # The cursor starts on cell (size 1, player 0), which the singletons
        # filled, and advances before every draw. It only stops on cells of
        # sizes 1 to n - 1, so a draw is never the empty or grand coalition.
        while generated < self._total_num_splits:
            if self._update_m_position():
                return
            pair = list(self._yield_complementary_pair(self._next_initial_split()))
            if pair:
                generated += len(pair)
                stale_attempts = 0
                yield from pair
                continue
            stale_attempts += 1
            if stale_attempts >= max_stale:
                return

    def begin_allocation(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
    ) -> None:
        """Estimate Neyman allocation after phase-one payoffs are available."""
        if self._m_counts is None:
            msg = "M matrix must be initialized before allocation."
            raise RuntimeError(msg)
        self._c_matrix = build_c_matrix_from_pairs(self._m_counts, masks, payoffs)
        self._c_squared = build_c_squared_from_pairs(self._m_counts, masks, payoffs)
        remaining_pairs = (self._total_num_splits - len(masks)) / 2
        if remaining_pairs < 0 or int(remaining_pairs) != remaining_pairs:
            msg = "Remaining samples for Neyman allocation must be an even integer."
            raise RuntimeError(msg)
        self._estimate_m_hat(int(remaining_pairs))
        self._step = _NeymanStep.NEYMAN_ALLOCATION
        self._position_j = self._allocation_start_index()
        self._position_i = 0

    def sample_allocation_masks(self) -> Iterator[CoalitionMask]:
        """Yield phase-two masks according to the estimated Neyman allocation."""
        if self._step != _NeymanStep.NEYMAN_ALLOCATION:
            msg = "Neyman allocation has not been initialized."
            raise RuntimeError(msg)
        if self._m_hat is None:
            msg = "M_hat must be initialized before allocation sampling."
            raise RuntimeError(msg)

        m_hat = self._m_hat
        stale_attempts = 0
        while (present := self._next_allocation_split()) is not None:
            pair = list(self._yield_complementary_pair(present))
            if pair:
                m_hat[self._position_j] -= 1
                stale_attempts = 0
                yield from pair
                continue
            # Allocation never exceeds the unseen pairs of a size, so repeated
            # draws only slow down near exhaustion; the guard is a safety net.
            stale_attempts += 1
            if stale_attempts >= 8 * math.comb(self._num_players, self._position_j):
                m_hat[self._position_j] = 0
                stale_attempts = 0

    def _estimate_m_hat(self, remaining_pairs: int) -> None:
        m_counts = self._m_counts
        c_matrix = self._c_matrix
        c_squared = self._c_squared
        if m_counts is None or c_matrix is None or c_squared is None:
            msg = "Neyman allocation inputs must be initialized."
            raise RuntimeError(msg)
        m_matrix = m_counts.astype(np.float64)
        m_small = m_matrix - 1.0
        sigma_squared = np.zeros_like(m_matrix, dtype=np.float64)
        valid = m_small > 0
        sigma_squared[valid] = (
            c_squared[valid] - np.square(c_matrix[valid]) / m_matrix[valid]
        ) / m_small[valid]
        sigma_squared = np.clip(sigma_squared, a_min=0.0, a_max=None)

        left = self._allocation_start_index()
        right = self._num_players
        coalition_indices = np.arange(left, right, dtype=np.int64)
        complement_indices = np.array(
            [
                self._complement_column(int(size), self._num_players)
                for size in coalition_indices
            ],
            dtype=np.int64,
        )
        k_left = coalition_indices.astype(np.float64)
        k_right = complement_indices.astype(np.float64)
        sigma_left = sigma_squared[:, coalition_indices]
        sigma_right = sigma_squared[:, complement_indices]
        inner = np.sqrt(
            np.sum(sigma_left / (k_left + 1.0), axis=0)
            + np.sum(sigma_right / (k_right + 1.0), axis=0)
        )
        capacity = self._unseen_pairs_by_size()[left:right]
        allocation = np.zeros(right - left, dtype=np.int64)
        remaining = min(remaining_pairs, int(capacity.sum()))
        while remaining > 0:
            # Water-filling: split the remaining pairs in proportion to the Neyman
            # weights among sizes that still have unseen pairs; zero variance
            # everywhere falls back to splitting by remaining capacity.
            spare = capacity - allocation
            active = spare > 0
            weights = np.where(active, inner, 0.0)
            if weights.sum() <= 0:
                weights = spare.astype(np.float64)
            share = np.floor(remaining * weights / weights.sum()).astype(np.int64)
            if share.sum() == 0:
                share[int(np.argmax(weights))] = 1
            share = np.minimum(share, spare)
            allocation += share
            remaining -= int(share.sum())
        m_hat = np.zeros(m_counts.shape[0], dtype=np.float64)
        m_hat[left:right] = allocation
        self._m_hat = m_hat
