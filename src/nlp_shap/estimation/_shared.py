"""Shared estimator base class and sampling utilities.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_shared.py]]
"""

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..protocols.estimand import EstimandAggregator
from ..protocols.plugin import PluginOptions

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class _SnapshotEstimator:
    """Plugin boilerplate shared by the built-in estimators."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context
        self._snapshot: ConversationSnapshot | None = None

    def bind_snapshot(self, snapshot: ConversationSnapshot) -> None:
        """Attach the conversation snapshot under explanation."""
        self._snapshot = snapshot

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Aggregate coalition payoffs with the selected estimand plugin."""
        return aggregator.aggregate([mask.present for mask in masks], payoffs)


def present_to_mask_int(present: Sequence[bool]) -> int:
    """Encode a coalition presence vector as a little-endian bitmask."""
    value = 0
    for index, is_present in enumerate(present):
        if is_present:
            value |= 1 << index
    return value


def mask_int_to_present(mask_int: int, num_players: int) -> tuple[bool, ...]:
    """Decode a coalition bitmask into presence flags."""
    return tuple(bool((mask_int >> index) & 1) for index in range(num_players))


def mc_coalition_budget(num_players: int) -> int:
    """Return the Monte Carlo coalition budget including empty and grand."""
    if num_players < 1:
        msg = "num_players must be at least 1"
        raise ValueError(msg)
    return 1 << num_players


def complementary_coalition_budget(num_players: int) -> int:
    """Return the complementary sampling budget over the full coalition table.

    The ``(N, empty)`` pair is always sampled first and counts toward the budget.
    """
    if num_players < 1:
        msg = "num_players must be at least 1"
        raise ValueError(msg)
    return 1 << num_players


def budget_fraction(
    num_players: int, fraction: float | None, evaluations: int | None
) -> float:
    """Return the ``budget_fraction`` the samplers receive for a run's budget.

    An absolute budget of ``evaluations`` coalitions becomes
    ``evaluations / 2^n``, capped at 1. Dividing by a power of two is exact in
    floating point, so ``int(2^n * fraction)`` in the count helpers gives back
    exactly ``evaluations`` before each sampler applies its floor and ceiling.
    Without ``evaluations`` the budget is ``fraction``, the full table when that
    is unset too.
    """
    if evaluations is None:
        return 1.0 if fraction is None else fraction
    return min(1.0, evaluations / (1 << num_players))


def compute_mc_num_samples(
    num_players: int,
    budget_fraction: float,
    include_minimal_masks: bool,
) -> int:
    """Return the Monte Carlo sample count capped by the coalition budget.

    The grand and empty coalitions are always sampled and count toward the
    budget, so the count is at least 2 (``n + 2`` with minimal masks).
    """
    maximum = mc_coalition_budget(num_players)
    minimum = num_players + 2 if include_minimal_masks else 2
    target = int(maximum * budget_fraction)
    return min(max(minimum, target), maximum)


def compute_complementary_num_splits(
    num_players: int,
    budget_fraction: float,
    include_minimal_masks: bool,
) -> int:
    """Return an even complementary mask budget derived from ``budget_fraction``.

    The floor is ``2n + 2`` masks (capped by the table size) so the ``(N, empty)``
    pair and all ``n`` singleton pairs ``({i}, N - {i})`` always fit; with fewer,
    a player's size-1 stratum stays empty and symmetry and efficiency break.
    """
    del include_minimal_masks
    maximum = complementary_coalition_budget(num_players)
    minimum = min(2 * num_players + 2, maximum)
    target = int(maximum * budget_fraction)
    # An odd target rounds down to whole pairs; the floor and the ceiling are
    # even for every n >= 1, so the clamped count stays even.
    return min(max(minimum, target - target % 2), maximum)


def iter_minimal_masks(num_players: int) -> Iterator[tuple[bool, ...]]:
    """Yield the empty coalition followed by each singleton coalition."""
    yield (False,) * num_players
    yield from iter_singletons(num_players)


def iter_singletons(num_players: int) -> Iterator[tuple[bool, ...]]:
    """Yield the singleton coalitions ``{0}``, ``{1}``, ... in player order."""
    for index in range(num_players):
        present = [False] * num_players
        present[index] = True
        yield tuple(present)


def random_present(
    rng: np.random.Generator,
    num_players: int,
    true_count: int | None = None,
    include_index: int | None = None,
) -> tuple[bool, ...]:
    """Sample a random coalition, optionally fixing size and one included player."""
    if true_count is None:
        bits = rng.integers(0, 2, size=num_players, dtype=np.int8)
        return tuple(bool(value) for value in bits)

    if include_index is not None:
        other_count = num_players - 1
        adjusted_count = true_count - 1
        if adjusted_count < 0 or adjusted_count > other_count:
            msg = "true_count incompatible with include_index"
            raise ValueError(msg)
        other_indices = [
            index for index in range(num_players) if index != include_index
        ]
        chosen = rng.choice(other_indices, size=adjusted_count, replace=False)
        present = [False] * num_players
        present[include_index] = True
        for index in chosen:
            present[int(index)] = True
        return tuple(present)

    if true_count < 0 or true_count > num_players:
        msg = "true_count out of range"
        raise ValueError(msg)
    chosen = rng.choice(num_players, size=true_count, replace=False)
    present = [False] * num_players
    for index in chosen:
        present[int(index)] = True
    return tuple(present)


def complementary_base_present(
    rng: np.random.Generator,
    num_players: int,
    include_index: int,
) -> tuple[bool, ...]:
    """Sample a complementary base coalition with one fixed included player."""
    true_count = math.ceil(num_players / 3)
    return random_present(
        rng,
        num_players,
        true_count=true_count,
        include_index=include_index,
    )


def is_empty_or_grand(present: Sequence[bool]) -> bool:
    """Return whether the coalition is empty or includes every player."""
    coalition_size = sum(present)
    return coalition_size == 0 or coalition_size == len(present)


def increment_m_counts(m_counts: np.ndarray, present: Sequence[bool]) -> None:
    """Increment complementary M-matrix counts for one coalition."""
    coalition_size = sum(present)
    if coalition_size == 0:
        m_counts[:, 0] += 1
        return
    for index, is_present in enumerate(present):
        if is_present:
            m_counts[index, coalition_size] += 1


def accumulate_complementary_contribution(
    c_matrix: np.ndarray,
    s_present: Sequence[bool],
    ns_present: Sequence[bool],
    contribution: float,
    power: int = 1,
) -> None:
    """Accumulate one complementary pair contribution into matrix C.

    With ``power=1`` the coalition side adds ``contribution`` and the complement
    side subtracts it (matrix C). With ``power=2`` both sides add
    ``contribution**2``, the squared sums used by Neyman variance estimates.
    """
    value = contribution**power
    complement_value = -value if power == 1 else value
    for present, signed in ((s_present, value), (ns_present, complement_value)):
        size = sum(present)
        if size == 0:
            c_matrix[:, 0] += signed
            continue
        for index, is_present in enumerate(present):
            if is_present:
                c_matrix[index, size] += signed


def aggregate_complementary_shapley(
    m_counts: np.ndarray,
    c_matrix: np.ndarray,
) -> list[float]:
    """Aggregate complementary M/C matrices into Shapley-style attributions."""
    m_work = m_counts[:, 1:]
    c_work = c_matrix[:, 1:]
    num_players = m_counts.shape[0]
    values = np.zeros(num_players, dtype=np.float64)
    for player_index in range(num_players):
        counts = m_work[player_index]
        contributions = c_work[player_index]
        ratio = np.zeros_like(counts, dtype=np.float64)
        positive = counts > 0
        ratio[positive] = contributions[positive] / counts[positive]
        values[player_index] = ratio.sum() / num_players
    return values.tolist()


def build_m_counts_from_masks(
    masks: Sequence[CoalitionMask],
    num_players: int,
) -> np.ndarray:
    """Rebuild the complementary M-matrix from an arbitrary mask list."""
    m_counts = np.zeros((num_players, num_players + 1), dtype=np.int64)
    for mask in masks:
        increment_m_counts(m_counts, mask.present)
    return m_counts


def pair_masks_by_complement(
    masks: Sequence[CoalitionMask],
    payoffs: Sequence[float],
) -> list[tuple[tuple[bool, ...], tuple[bool, ...], float, float]]:
    """Group masks into ``(S, N - S, v(S), v(N - S))`` tuples regardless of order.

    Raises:
        ValueError: If a mask has no complement in ``masks``.
    """
    lookup = {mask.present: payoff for mask, payoff in zip(masks, payoffs, strict=True)}
    pairs: list[tuple[tuple[bool, ...], tuple[bool, ...], float, float]] = []
    consumed: set[tuple[bool, ...]] = set()
    for present, payoff in lookup.items():
        if present in consumed:
            continue
        complement = tuple(not value for value in present)
        if complement not in lookup:
            msg = f"mask {present!r} has no complementary pair"
            raise ValueError(msg)
        consumed.add(present)
        consumed.add(complement)
        pairs.append((present, complement, payoff, lookup[complement]))
    return pairs


def build_c_matrix_from_pairs(
    m_counts: np.ndarray,
    masks: Sequence[CoalitionMask],
    payoffs: Sequence[float],
) -> np.ndarray:
    """Build matrix C from complementary mask pairs and coalition payoffs.

    Masks are paired by complement, so their order does not matter.
    """
    c_matrix = np.zeros_like(m_counts, dtype=np.float64)
    for s_present, ns_present, v_s, v_ns in pair_masks_by_complement(masks, payoffs):
        accumulate_complementary_contribution(
            c_matrix,
            s_present,
            ns_present,
            v_s - v_ns,
        )
    return c_matrix


def build_c_squared_from_pairs(
    m_counts: np.ndarray,
    masks: Sequence[CoalitionMask],
    payoffs: Sequence[float],
) -> np.ndarray:
    """Build the squared complementary matrix used by Neyman allocation."""
    c_squared = np.zeros_like(m_counts, dtype=np.float64)
    for s_present, ns_present, v_s, v_ns in pair_masks_by_complement(masks, payoffs):
        accumulate_complementary_contribution(
            c_squared,
            s_present,
            ns_present,
            v_s - v_ns,
            power=2,
        )
    return c_squared
