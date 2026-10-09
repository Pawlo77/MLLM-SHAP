"""Tests for complementary pair sampling.

@see [[src/nlp_shap/estimation/complementary.py]]
"""

import numpy as np
import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation.complementary import ComplementaryEstimator
from nlp_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from nlp_shap.estimation.estimands._weights import aggregate_shapley_exact
from tests.estimation._games import additive_payoffs, full_table


def test_complementary_pairs_are_negations() -> None:
    """Each emitted pair consists of a mask and its bitwise complement."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    masks = list(
        ComplementaryEstimator().sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=True,
            seed=1,
        )
    )
    assert len(masks) % 2 == 0
    for index in range(0, len(masks), 2):
        left = masks[index].present
        right = masks[index + 1].present
        assert tuple(not value for value in left) == right


def test_complementary_budget_is_even_and_capped() -> None:
    """Complementary sampling respects the even budget cap."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = list(
        ComplementaryEstimator().sample_masks(
            player_set,
            budget_fraction=0.4,
            include_minimal_masks=True,
            seed=2,
        )
    )
    assert len(masks) % 2 == 0
    assert len(masks) <= 14


def test_complementary_full_table_matches_exact_shapley() -> None:
    """On the full 2-player table CC aggregation equals exact Shapley values."""
    masks = [CoalitionMask.from_sequence(present) for present in full_table(2)]
    # v(empty)=0, v({0})=2, v({1})=1, v(N)=1: phi = (1, 0) by hand.
    payoffs = [0.0, 2.0, 1.0, 1.0]
    values = ComplementaryEstimator().estimate_attributions(
        masks, payoffs, ShapleyAggregator()
    )
    assert values == pytest.approx(aggregate_shapley_exact(full_table(2), payoffs))
    assert values == pytest.approx([1.0, 0.0])


def test_complementary_rejects_unpaired_mask() -> None:
    """A mask without its complement cannot be aggregated."""
    estimator = ComplementaryEstimator()
    masks = (CoalitionMask.from_sequence((True, False)),)
    with pytest.raises(ValueError, match="complementary pair"):
        _ = estimator.estimate_attributions(masks, [1.0], ShapleyAggregator())


def test_complementary_rejects_non_shapley_estimand() -> None:
    """Complementary contributions only estimate the Shapley value."""
    masks = (
        CoalitionMask.from_sequence((True, False)),
        CoalitionMask.from_sequence((False, True)),
    )
    with pytest.raises(ValueError, match="shapley"):
        _ = ComplementaryEstimator().estimate_attributions(
            masks, [1.0, 0.0], BanzhafAggregator()
        )


def test_complementary_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert ComplementaryEstimator().name == "complementary"


def test_complementary_full_budget_matches_exact_on_symmetric_game() -> None:
    """With the (N, empty) pair included, CC at full budget is efficient."""
    num_players = 4
    player_set = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    estimator = ComplementaryEstimator()
    masks = list(estimator.sample_masks(player_set, 1.0, False, seed=3))
    assert len(masks) == 2**num_players
    payoffs = [sum(m.present) / num_players for m in masks]
    values = estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
    assert sum(values) == pytest.approx(1.0, abs=1e-9)
    assert values == pytest.approx([0.25] * num_players, abs=1e-9)


def test_complementary_estimate_is_order_independent() -> None:
    """Aggregation pairs masks by complement, not by yield order."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    estimator = ComplementaryEstimator()
    masks = list(estimator.sample_masks(player_set, 1.0, False, seed=0))
    coefficients = [1.0, 2.0, 3.0]
    payoffs = additive_payoffs([m.present for m in masks], coefficients)
    shuffled = list(zip(masks, payoffs, strict=True))
    np.random.default_rng(5).shuffle(shuffled)
    masks_s = [m for m, _ in shuffled]
    payoffs_s = [p for _, p in shuffled]
    fresh = ComplementaryEstimator()  # no sampling state at all
    values = fresh.estimate_attributions(masks_s, payoffs_s, ShapleyAggregator())
    assert values == pytest.approx(coefficients)


def test_complementary_partial_budget_efficiency_unbiased_on_additive_game() -> None:
    """Zero-filled size-1 strata used to inflate the attribution sum (+1.7)."""
    n = 8
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(n)))
    weights = np.linspace(0.2, 1.0, n)
    gaps = []
    for seed in range(100):
        estimator = ComplementaryEstimator()
        masks = list(estimator.sample_masks(players, 0.5, False, seed))
        payoffs = [float(weights @ np.array(m.present)) for m in masks]
        values = estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
        gaps.append(sum(values) - weights.sum())
    assert abs(np.mean(gaps)) < 0.25


def test_complementary_covers_every_size_one_stratum_at_half_budget() -> None:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(8)))
    for seed in range(20):
        estimator = ComplementaryEstimator()
        list(estimator.sample_masks(players, 0.5, False, seed))
        assert estimator.m_counts is not None
        assert estimator.m_counts[:, 1].min() > 0


def test_complementary_yields_singleton_pairs_right_after_boundary_pair() -> None:
    players = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = [
        m.present for m in ComplementaryEstimator().sample_masks(players, 1.0, False, 0)
    ]
    singletons = [tuple(index == player for index in range(4)) for player in range(4)]
    expected = [
        mask
        for single in singletons
        for mask in (single, tuple(not value for value in single))
    ]
    assert masks[2:10] == expected
    assert len(set(masks)) == 16


def test_complementary_singleton_pairs_fit_a_small_budget() -> None:
    """The floor budget fits ``(N, empty)`` and every singleton pair."""
    players = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = [
        m.present for m in ComplementaryEstimator().sample_masks(players, 0.1, False, 0)
    ]
    # Budget: 2n + 2 = 10 masks -> (N, empty) and all four singleton pairs.
    singletons = [tuple(index == player for index in range(4)) for player in range(4)]
    assert masks[2:] == [
        mask
        for single in singletons
        for mask in (single, tuple(not value for value in single))
    ]
