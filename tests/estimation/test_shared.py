"""Tests for shared approximate-estimator utilities.

@see [[src/nlp_shap/estimation/_shared.py]]
"""

from collections.abc import Callable

import numpy as np
import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.estimation._shared import (
    aggregate_complementary_shapley,
    budget_fraction,
    build_c_matrix_from_pairs,
    build_m_counts_from_masks,
    complementary_coalition_budget,
    compute_complementary_num_splits,
    compute_mc_num_samples,
    increment_m_counts,
    mc_coalition_budget,
    present_to_mask_int,
    random_present,
)
from nlp_shap.estimation.estimands._weights import aggregate_shapley_exact
from tests.estimation._games import full_table


def test_compute_mc_num_samples_respects_budget_fraction() -> None:
    """Monte Carlo sample counts scale with budget fraction."""
    assert compute_mc_num_samples(4, 0.5, include_minimal_masks=False) == 8
    assert compute_mc_num_samples(4, 1.0, include_minimal_masks=False) == 16


def test_compute_complementary_num_splits_is_even() -> None:
    """Complementary budgets are even and respect the minimum pair count."""
    assert compute_complementary_num_splits(4, 0.5, include_minimal_masks=True) == 10
    assert compute_complementary_num_splits(5, 0.5, include_minimal_masks=True) == 16
    assert compute_complementary_num_splits(5, 0.7, include_minimal_masks=True) == 22
    assert compute_complementary_num_splits(2, 0.1, include_minimal_masks=True) == 4


def test_increment_m_counts_tracks_coalition_sizes() -> None:
    """M-matrix counts increment on player and coalition-size axes."""
    counts = np.zeros((2, 3), dtype=np.int64)
    increment_m_counts(counts, (True, False))
    assert counts[0, 1] == 1
    assert counts[1, 0] == 0


def test_build_c_matrix_from_pairs_requires_complements() -> None:
    """Non-complementary pairs are rejected when building matrix C."""
    counts = np.zeros((2, 3), dtype=np.int64)
    masks = (
        CoalitionMask.from_sequence((True, False)),
        CoalitionMask.from_sequence((True, False)),
    )
    with pytest.raises(ValueError, match="complementary pair"):
        _ = build_c_matrix_from_pairs(counts, masks, [1.0, 2.0])


def test_aggregate_complementary_shapley_matches_exact_on_full_table() -> None:
    """M/C matrices built from the full table aggregate to exact Shapley values."""
    num_players = 3
    table = full_table(num_players)
    # Non-additive game: unit per player plus a bonus when players 0 and 2 meet.
    payoffs = [sum(mask) + (2.5 if mask[0] and mask[2] else 0.0) for mask in table]
    masks = [CoalitionMask.from_sequence(mask) for mask in table]
    m_counts = build_m_counts_from_masks(masks, num_players)
    c_matrix = build_c_matrix_from_pairs(m_counts, masks, payoffs)

    values = aggregate_complementary_shapley(m_counts, c_matrix)

    assert values == pytest.approx(aggregate_shapley_exact(table, payoffs))
    assert values == pytest.approx([2.25, 1.0, 2.25])


def test_random_present_honors_include_index() -> None:
    """Random coalition generation can force one player to be present."""
    rng = np.random.default_rng(0)
    present = random_present(rng, 4, true_count=2, include_index=1)
    assert present[1] is True
    assert sum(present) == 2


def test_present_to_mask_int_round_trip() -> None:
    """Coalition bitmasks round-trip through present vectors."""
    present = (True, False, True)
    assert present_to_mask_int(present) == 5


@pytest.mark.parametrize(
    "budget", [mc_coalition_budget, complementary_coalition_budget]
)
def test_coalition_budgets_are_the_table_size(budget: Callable[[int], int]) -> None:
    assert [budget(n) for n in range(1, 7)] == [2, 4, 8, 16, 32, 64]
    with pytest.raises(ValueError, match=r"^num_players must be at least 1$"):
        budget(0)


@pytest.mark.parametrize(
    ("num_players", "fraction", "minimal", "expected"),
    [
        # min(max(floor, int(2^n * fraction)), 2^n); floor 2, or n + 2 with
        # minimal masks.
        (1, 0.3, False, 2),  # int(0.6) = 0 -> floor 2
        (1, 0.3, True, 2),  # floor 3 is capped by the table size 2
        (2, 0.3, True, 4),  # int(1.2) = 1 -> floor 4 = 2^2
        (3, 0.3, False, 2),  # int(2.4) = 2
        (3, 0.3, True, 5),  # floor 3 + 2
        (4, 0.3, False, 4),  # int(4.8) = 4
        (4, 0.3, True, 6),  # floor 4 + 2
        (5, 0.3, True, 9),  # int(9.6) = 9 above floor 7
        (6, 0.3, False, 19),  # int(19.2) = 19, odd counts are kept
        (6, 1.0, True, 64),
    ],
)
def test_compute_mc_num_samples_exact_counts(
    num_players: int, fraction: float, minimal: bool, expected: int
) -> None:
    assert compute_mc_num_samples(num_players, fraction, minimal) == expected


@pytest.mark.parametrize(
    ("num_players", "fraction", "expected"),
    [
        # Floor min(2n + 2, 2^n), ceiling 2^n, odd targets rounded down.
        (1, 0.3, 2),  # floor min(4, 2) = 2
        (1, 1.0, 2),
        (2, 0.3, 4),  # floor min(6, 4) = 4 = 2^2
        (3, 0.3, 8),  # floor min(8, 8) = 8
        (4, 0.3, 10),  # int(4.8) = 4 -> floor 10
        (4, 0.6875, 10),  # int(11.0) = 11 -> 10, the floor
        (4, 0.8125, 12),  # int(13.0) = 13 -> 12
        (4, 1.0, 16),
        (5, 0.3, 12),  # int(9.6) = 9 -> floor 12
        (5, 0.53125, 16),  # int(17.0) = 17 -> 16
        (6, 0.1, 14),  # int(6.4) = 6 -> floor 14
        (6, 0.3, 18),  # int(19.2) = 19 -> 18
        (6, 0.99, 62),  # int(63.36) = 63 -> 62
        (6, 1.0, 64),
    ],
)
def test_compute_complementary_num_splits_exact_counts(
    num_players: int, fraction: float, expected: int
) -> None:
    for minimal in (False, True):
        assert (
            compute_complementary_num_splits(num_players, fraction, minimal) == expected
        )


def test_random_present_seeded_draws() -> None:
    """The draws for a fixed seed are pinned, so a change of RNG call shows."""
    rng = np.random.default_rng(7)
    assert [random_present(rng, 5) for _ in range(3)] == [
        (True, False, True, True, True),
        (True, False, False, True, True),
        (False, True, False, True, False),
    ]
    rng = np.random.default_rng(7)
    assert [random_present(rng, 5, true_count=2) for _ in range(3)] == [
        (False, False, False, True, True),
        (False, False, True, True, False),
        (False, True, False, True, False),
    ]
    rng = np.random.default_rng(7)
    assert [
        random_present(rng, 5, true_count=3, include_index=4) for _ in range(3)
    ] == [
        (False, False, True, True, True),
        (False, False, True, True, True),
        (True, False, True, False, True),
    ]


def test_random_present_unrestricted_draws_are_uniform() -> None:
    """Each player is present with probability 1/2 and every coalition occurs."""
    rng = np.random.default_rng(0)
    draws = np.array([random_present(rng, 4) for _ in range(4000)])
    assert draws.dtype == np.bool_
    assert draws.mean(axis=0) == pytest.approx([0.5] * 4, abs=0.03)
    assert len({tuple(row) for row in draws.tolist()}) == 16


def test_random_present_fixed_size_draws_are_uniform() -> None:
    """Size-k draws hold k distinct players, each present with probability k/n."""
    rng = np.random.default_rng(1)
    draws = np.array([random_present(rng, 5, true_count=2) for _ in range(4000)])
    assert set(draws.sum(axis=1).tolist()) == {2}
    assert draws.mean(axis=0) == pytest.approx([2 / 5] * 5, abs=0.03)


def test_random_present_with_include_index_draws_the_rest_uniformly() -> None:
    """The fixed player is always present; the other k - 1 of n - 1 are uniform."""
    rng = np.random.default_rng(2)
    draws = np.array([
        random_present(rng, 5, true_count=3, include_index=1) for _ in range(4000)
    ])
    assert set(draws.sum(axis=1).tolist()) == {3}
    assert draws[:, 1].all()
    others = np.delete(draws, 1, axis=1)
    assert others.mean(axis=0) == pytest.approx([2 / 4] * 4, abs=0.03)


def test_random_present_boundary_sizes() -> None:
    rng = np.random.default_rng(0)
    assert random_present(rng, 3, true_count=0) == (False, False, False)
    assert random_present(rng, 3, true_count=3) == (True, True, True)
    assert random_present(rng, 3, true_count=1, include_index=2) == (
        False,
        False,
        True,
    )
    assert random_present(rng, 3, true_count=3, include_index=0) == (True,) * 3
    with pytest.raises(ValueError, match=r"^true_count out of range$"):
        random_present(rng, 3, true_count=-1)
    with pytest.raises(ValueError, match=r"^true_count out of range$"):
        random_present(rng, 3, true_count=4)
    with pytest.raises(
        ValueError, match=r"^true_count incompatible with include_index$"
    ):
        random_present(rng, 3, true_count=0, include_index=1)
    with pytest.raises(
        ValueError, match=r"^true_count incompatible with include_index$"
    ):
        random_present(rng, 3, true_count=4, include_index=1)


@pytest.mark.parametrize("num_players", range(1, 7))
def test_absolute_budget_converts_to_exact_sampler_counts(num_players: int) -> None:
    table = 2**num_players
    floor = min(2 * num_players + 2, table)
    for evaluations in range(1, table + 4):
        fraction = budget_fraction(num_players, None, evaluations)
        assert compute_mc_num_samples(num_players, fraction, False) == min(
            max(2, evaluations), table
        )
        assert compute_complementary_num_splits(num_players, fraction, False) == min(
            max(floor, evaluations - evaluations % 2), table
        )


def test_absolute_budget_is_exact_for_many_players() -> None:
    # evaluations / 2^n is exact in floating point, so 2^n * fraction gives the
    # count back even where 2^n is far beyond any fraction a user could type.
    for num_players in (40, 300, 1000):
        fraction = budget_fraction(num_players, None, 2001)
        assert compute_mc_num_samples(num_players, fraction, False) == 2001
        assert compute_complementary_num_splits(num_players, fraction, False) == max(
            2 * num_players + 2, 2000
        )


def test_fraction_budget_is_unchanged() -> None:
    assert budget_fraction(5, 0.3, None) == 0.3
    assert budget_fraction(5, None, None) == 1.0
    # More evaluations than the table holds is the full table.
    assert budget_fraction(3, None, 100) == 1.0
