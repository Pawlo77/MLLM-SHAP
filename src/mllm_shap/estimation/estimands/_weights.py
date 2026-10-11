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
    """Estimate per-player values from observed marginal contributions.

    Each observed pair ``(S, S | {i})`` with ``i not in S`` contributes the
    marginal ``v(S | {i}) - v(S)`` to the stratum ``|S|``. Under Shapley weights
    every stratum carries weight ``1/n``; under Banzhaf weights stratum ``k``
    carries ``C(n-1, k) / 2^(n-1)``. The estimate is the probability-weighted
    mean of the per-stratum mean marginals, renormalized over the strata that
    have observations. On a complete table it equals the exact value. A player
    with no observed marginal gets 0 and a WARNING is logged.

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

    values = [0.0] * num_players
    unobserved: list[int] = []
    for player_index in range(num_players):
        player_bit = 1 << player_index
        stratum_sum = np.zeros(num_players, dtype=np.float64)
        stratum_count = np.zeros(num_players, dtype=np.int64)
        for mask_int, payoff in payoff_lookup.items():
            if mask_int & player_bit:
                continue
            extended_payoff = payoff_lookup.get(mask_int | player_bit)
            if extended_payoff is None:
                continue
            size = mask_int.bit_count()
            stratum_sum[size] += extended_payoff - payoff
            stratum_count[size] += 1

        observed = stratum_count > 0
        if not observed.any():
            unobserved.append(player_index)
            continue
        stratum_mean = stratum_sum[observed] / stratum_count[observed]
        # p_k = weight(k, n) * C(n-1, k) is the semivalue's probability of stratum
        # k (1/n each for Shapley, C(n-1, k)/2^(n-1) for Banzhaf). With every
        # stratum observed this is an unbiased stratified mean; renormalizing over
        # the observed strata drops the missing ones, which biases the estimate
        # whenever their mean marginal differs from the observed ones.
        probabilities = np.array(
            [
                weight_fn(int(size), num_players)
                * math.comb(num_players - 1, int(size))
                for size in np.flatnonzero(observed)
            ],
            dtype=np.float64,
        )
        values[player_index] = float(
            (probabilities * stratum_mean).sum() / probabilities.sum()
        )

    if unobserved:
        logger.warning(
            "%d of %d players have no observed marginal (no sampled pair S, "
            "S | {i}); their values are 0. Raise budget.fraction or set "
            "include_minimal_masks. Players: %s",
            len(unobserved),
            num_players,
            unobserved,
        )
    return values
