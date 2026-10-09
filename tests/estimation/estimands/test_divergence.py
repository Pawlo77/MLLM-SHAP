"""Tests that Shapley and Banzhaf diverge on identical coalition samples.

@see [[knowledge/architecture/estimation]]
"""

from itertools import product

import pytest

from nlp_shap.domain.estimands import Estimand
from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator


def _majority_game_payoff(mask: tuple[bool, ...]) -> float:
    """Return 1 when at least two players are present, else 0."""
    return 1.0 if sum(mask) >= 2 else 0.0


def test_shapley_and_banzhaf_differ_on_majority_game() -> None:
    """Identical samples yield different attributions for non-additive games."""
    num_players = 3
    masks = [tuple(bits) for bits in product([False, True], repeat=num_players)]
    payoffs = [_majority_game_payoff(mask) for mask in masks]

    shapley = ShapleyAggregator().aggregate(masks, payoffs)
    banzhaf = BanzhafAggregator().aggregate(masks, payoffs)

    assert shapley != banzhaf
    assert all(value == pytest.approx(2 / 6) for value in shapley)
    assert all(value == pytest.approx(0.5) for value in banzhaf)


def test_additive_game_matches_for_both_estimands() -> None:
    """Linear games recover coefficients under either estimand."""
    num_players = 2
    coefficients = (1.0, 2.0)
    masks = [tuple(bits) for bits in product([False, True], repeat=num_players)]
    payoffs = [
        sum(coef for coef, present in zip(coefficients, mask, strict=True) if present)
        for mask in masks
    ]

    shapley = ShapleyAggregator().aggregate(masks, payoffs)
    banzhaf = BanzhafAggregator().aggregate(masks, payoffs)

    assert shapley == pytest.approx(list(coefficients))
    assert banzhaf == pytest.approx(list(coefficients))


def test_aggregators_expose_estimand_labels() -> None:
    """Each aggregator reports its estimand enum."""
    assert ShapleyAggregator().estimand is Estimand.SHAPLEY
    assert BanzhafAggregator().estimand is Estimand.BANZHAF
