"""Shapley and Banzhaf axioms on complete characteristic-function tables.

@see [[knowledge/architecture/estimation]]
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation._shared import compute_complementary_num_splits
from nlp_shap.estimation.complementary import ComplementaryEstimator
from nlp_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from nlp_shap.estimation.estimands._weights import (
    aggregate_from_marginals,
    banzhaf_weight,
    shapley_weight,
)
from nlp_shap.estimation.neyman import NeymanEstimator
from tests.estimation._games import additive_payoffs, full_table, random_game

Game = tuple[int, list[tuple[bool, ...]], list[float]]


@given(random_game())
@settings(max_examples=60)
def test_shapley_efficiency(game: Game) -> None:
    _, masks, payoffs = game
    values = ShapleyAggregator().aggregate(masks, payoffs)
    assert sum(values) == pytest.approx(payoffs[-1] - payoffs[0], abs=1e-6)


@given(random_game(min_players=2))
@settings(max_examples=60)
def test_shapley_symmetry(game: Game) -> None:
    _, masks, payoffs = game
    # Make players 0 and 1 interchangeable: v depends on how many of them are present.
    lookup = dict(zip(masks, payoffs, strict=True))
    symmetric = [lookup[(min(m[0], m[1]), max(m[0], m[1]), *m[2:])] for m in masks]
    values = ShapleyAggregator().aggregate(masks, symmetric)
    assert values[0] == pytest.approx(values[1], abs=1e-6)


@given(random_game(min_players=2))
@settings(max_examples=60)
def test_shapley_dummy(game: Game) -> None:
    _, masks, payoffs = game
    lookup = dict(zip(masks, payoffs, strict=True))
    # Player 0 never changes the payoff.
    dummy = [lookup[(False, *m[1:])] for m in masks]
    values = ShapleyAggregator().aggregate(masks, dummy)
    assert values[0] == pytest.approx(0.0, abs=1e-6)


@given(random_game(), st.data())
@settings(max_examples=40)
def test_shapley_additivity(
    left: Game,
    data: st.DataObject,
) -> None:
    _, masks, pay_left = left
    pay_right = data.draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            min_size=len(masks),
            max_size=len(masks),
        )
    )
    combined = [a + b for a, b in zip(pay_left, pay_right, strict=True)]
    aggregator = ShapleyAggregator()
    expected = [
        a + b
        for a, b in zip(
            aggregator.aggregate(masks, pay_left),
            aggregator.aggregate(masks, pay_right),
            strict=True,
        )
    ]
    assert aggregator.aggregate(masks, combined) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("num_players", [1, 2, 3, 5])
def test_additive_game_recovers_coefficients(num_players: int) -> None:
    coefficients = [float(index + 1) for index in range(num_players)]
    masks = full_table(num_players)
    payoffs = additive_payoffs(masks, coefficients)
    assert ShapleyAggregator().aggregate(masks, payoffs) == pytest.approx(coefficients)
    assert BanzhafAggregator().aggregate(masks, payoffs) == pytest.approx(coefficients)


@pytest.mark.parametrize("num_players", [4, 5, 6])
def test_partial_sample_with_every_stratum_matches_exact_semivalues(
    num_players: int,
) -> None:
    """v(S) = |S|^3 on a table missing coalitions but no (player, size) stratum.

    The marginal of stratum k is 3k^2 + 3k + 1 for every player, so the stratified
    estimate is exact: Shapley n^2, Banzhaf E[3k^2 + 3k + 1] with k ~ Bin(n-1, 1/2).
    Mixing up the stratum weights would give the other semivalue.
    """
    masks = full_table(num_players)
    # Drop coalitions holding players 0 and 1 with size in [2, n-2]; each
    # (player, size) stratum keeps a pair without player 1 (or with the grand).
    keep = [
        mask
        for mask in masks
        if not (mask[0] and mask[1] and 2 <= sum(mask) <= num_players - 2)
    ]
    assert len(keep) < len(masks)
    payoffs = [float(sum(mask)) ** 3 for mask in keep]
    k_mean = (num_players - 1) / 2
    k_square_mean = (num_players - 1) / 4 + k_mean**2
    banzhaf_exact = 3 * k_square_mean + 3 * k_mean + 1

    shapley = aggregate_from_marginals(keep, payoffs, shapley_weight)
    banzhaf = aggregate_from_marginals(keep, payoffs, banzhaf_weight)

    assert shapley == pytest.approx([float(num_players**2)] * num_players, abs=1e-9)
    assert banzhaf == pytest.approx([banzhaf_exact] * num_players, abs=1e-9)
    assert banzhaf_exact != pytest.approx(num_players**2)


def test_partial_sample_scale_matches_full_table() -> None:
    """Half table (player 0 absent, plus N) of v(S) = |S|^2 for 4 players.

    Player 0 is observed only in stratum 3 (marginal 16 - 9 = 7); players 1-3
    only in strata 0-2 (marginals 1, 3, 5, mean 3). The values keep the full
    table's scale: they sum to v(N) - v(empty) = 16, like the exact [4, 4, 4, 4].
    """
    masks = full_table(4)
    payoffs = [float(sum(m) ** 2) for m in masks]
    full = aggregate_from_marginals(masks, payoffs, shapley_weight)
    half = aggregate_from_marginals(
        [*masks[::2], masks[-1]], [*payoffs[::2], payoffs[-1]], shapley_weight
    )
    assert full == pytest.approx([4.0, 4.0, 4.0, 4.0], abs=1e-12)
    assert half == pytest.approx([7.0, 3.0, 3.0, 3.0], abs=1e-12)
    assert sum(half) == pytest.approx(16.0, abs=1e-12)


@pytest.mark.parametrize("estimator_cls", [ComplementaryEstimator, NeymanEstimator])
@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_complementary_floor_budget_is_symmetric_and_efficient(
    estimator_cls: type[ComplementaryEstimator],
    num_players: int,
) -> None:
    """At the floor budget every player gets its singleton pair, so v(S) = |S|."""
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    estimator = estimator_cls()
    masks = list(estimator.sample_masks(players, 0.01, False, 0))
    assert len(masks) == compute_complementary_num_splits(num_players, 0.01, False)
    payoffs = [float(sum(mask.present)) for mask in masks]
    if isinstance(estimator, NeymanEstimator):
        estimator.begin_allocation(masks, payoffs)
        assert list(estimator.sample_allocation_masks()) == []
    values = estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
    assert values == pytest.approx([1.0] * num_players, abs=1e-12)
    assert sum(values) == pytest.approx(float(num_players), abs=1e-12)
