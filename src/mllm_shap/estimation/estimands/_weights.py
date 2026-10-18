"""Shared helpers for estimand aggregation.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/estimands/test_weights.py]]
"""

import logging
import math
from collections.abc import Callable, Sequence

import numpy as np

from .._shared import present_to_mask_int

logger = logging.getLogger(__name__)


def is_complete_characteristic_table(masks: Sequence[Sequence[bool]]) -> bool:
    """Return whether masks include every coalition for n players."""
    num_players = len(masks[0])
    expected = 1 << num_players
    if len(masks) != expected:
        return False
    seen = {present_to_mask_int(mask) for mask in masks}
    return len(seen) == expected


def shapley_weight(coalition_size: int, num_players: int) -> float:
    """Return Shapley coalition weight k!(n-k-1)!/n!."""
    coalition_size_without_player = coalition_size
    return (
        math.factorial(coalition_size_without_player)
        * math.factorial(num_players - coalition_size_without_player - 1)
        / math.factorial(num_players)
    )


def banzhaf_weight(_coalition_size: int, num_players: int) -> float:
    """Return uniform Banzhaf coalition weight 1/2^(n-1)."""
    return 1.0 / float(2 ** (num_players - 1))


def aggregate_shapley_exact(
    masks: Sequence[Sequence[bool]],
    payoffs: Sequence[float],
) -> list[float]:
    """Compute Shapley values from a complete characteristic-function table."""
    if not masks:
        msg = "masks must not be empty"
        raise ValueError(msg)
    if len(masks) != len(payoffs):
        msg = "masks and payoffs must have the same length"
        raise ValueError(msg)

    array = np.asarray(masks, dtype=np.bool_)
    if array.ndim != 2:
        msg = "masks must be a 2D coalition table"
        raise ValueError(msg)

    num_features = int(array.shape[1])
    pay = np.asarray(payoffs, dtype=np.float64)
    shap_values = np.zeros(num_features, dtype=np.float64)
    indices = np.arange(num_features + 1, dtype=np.float64)
    indices[0] = 1.0
    factorials = np.cumprod(indices)
    power_of_2 = 2 ** np.arange(num_features, dtype=np.int64)
    subset_hashes = (array.astype(np.int64) * power_of_2).sum(axis=1)
    sorted_hashes = np.sort(subset_hashes)
    sort_idx = np.argsort(subset_hashes)
    sorted_outputs = pay[sort_idx]
    subset_sizes = array.sum(axis=1)

    for player_index in range(num_features):
        include_mask = array[:, player_index]
        included_outputs = pay[include_mask]
        included_hashes = subset_hashes[include_mask]
        excluded_hash = included_hashes - power_of_2[player_index]
        excluded_outputs = sorted_outputs[np.searchsorted(sorted_hashes, excluded_hash)]
        excluded_subset_sizes = subset_sizes[include_mask] - 1
        weights = (
            factorials[excluded_subset_sizes]
            * factorials[num_features - excluded_subset_sizes - 1]
            / factorials[num_features]
        )
        shap_values[player_index] = np.sum(
            weights * (included_outputs - excluded_outputs)
        )
    return shap_values.tolist()


def aggregate_from_marginals(
    masks: Sequence[Sequence[bool]],
    payoffs: Sequence[float],
    weight_fn: Callable[[int, int], float],
) -> list[float]:
    """Estimate per-player values from sampled coalitions, using every one.

    For player ``i`` and stratum ``l`` (coalition size without ``i``) the
    estimate is the mean payoff of the sampled coalitions that contain ``i`` and
    have size ``l + 1`` minus the mean payoff of those that exclude ``i`` and have
    size ``l`` (Stratified SVARM, Kolpaczki et al. 2024). The strata are combined
    with the semivalue's probabilities (``1/n`` for Shapley, ``C(n-1, l)/2^(n-1)``
    for Banzhaf), renormalized over the strata that have both means. Each sample
    updates every player, and a complete table gives the exact value. A player
    with no stratum observed on both sides gets 0 and a WARNING is logged.

    Args:
        masks: Coalition membership rows with shape ``(m, n)``.
        payoffs: Characteristic-function samples ``v(S)`` aligned with masks.
        weight_fn: Coalition weight callback receiving ``(coalition_size, n)``.

    Returns:
        Per-player attribution vector with length ``n``.

    Raises:
        ValueError: If masks are empty, ragged, or misaligned with payoffs.
    """
    if not masks:
        msg = "masks must not be empty"
        raise ValueError(msg)
    if len(masks) != len(payoffs):
        msg = "masks and payoffs must have the same length"
        raise ValueError(msg)

    num_players = len(masks[0])
    if num_players == 0:
        msg = "masks must include at least one player"
        raise ValueError(msg)
    if any(len(mask) != num_players for mask in masks):
        msg = "all masks must have the same number of players"
        raise ValueError(msg)

    payoff_lookup: dict[int, float] = {}
    for mask, payoff in zip(masks, payoffs, strict=True):
        payoff_lookup[present_to_mask_int(mask)] = payoff
    unique = np.array(
        [[(key >> i) & 1 for i in range(num_players)] for key in payoff_lookup],
        dtype=np.float64,
    )
    pay = np.fromiter(payoff_lookup.values(), dtype=np.float64)
    onehot = np.eye(num_players + 1)[unique.sum(axis=1).astype(np.int64)]
    with_sum, with_count = (unique * pay[:, None]).T @ onehot, unique.T @ onehot
    absent = 1.0 - unique
    without_sum, without_count = (absent * pay[:, None]).T @ onehot, absent.T @ onehot

    # Stratum l pairs the size-(l + 1) coalitions with i and the size-l without.
    valid = (with_count[:, 1:] > 0) & (without_count[:, :-1] > 0)
    difference = np.divide(
        with_sum[:, 1:], with_count[:, 1:], where=valid, out=np.zeros_like(valid, float)
    ) - np.divide(
        without_sum[:, :-1],
        without_count[:, :-1],
        where=valid,
        out=np.zeros_like(valid, float),
    )
    # p_l = weight(l, n) * C(n-1, l) is the semivalue's probability of stratum l.
    probabilities = np.array([
        weight_fn(size, num_players) * math.comb(num_players - 1, size)
        for size in range(num_players)
    ])
    mass = (valid * probabilities).sum(axis=1)
    unobserved = np.flatnonzero(mass == 0).tolist()
    values = np.divide(
        (valid * probabilities * difference).sum(axis=1),
        mass,
        where=mass > 0,
        out=np.zeros(num_players),
    )
    if unobserved:
        logger.warning(
            "%d of %d players have no stratum sampled with and without them; "
            "their values are 0. Raise the budget. Players: %s",
            len(unobserved),
            num_players,
            unobserved,
        )
    return [float(value) for value in values]
