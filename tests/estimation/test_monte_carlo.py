"""Tests for Monte Carlo coalition sampling.

@see [[src/nlp_shap/estimation/monte_carlo.py]]
"""

import logging
from itertools import product

import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator
from nlp_shap.estimation.exact import ExactEstimator
from nlp_shap.estimation.monte_carlo import MonteCarloEstimator


def _majority_payoffs(masks: tuple[tuple[bool, ...], ...]) -> list[float]:
    return [1.0 if sum(mask) >= 2 else 0.0 for mask in masks]


def test_mc_budget_cap_is_honoured() -> None:
    """Monte Carlo sampling never exceeds the configured budget."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=False,
            seed=7,
        )
    )
    assert len(masks) <= 4


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_mc_budget_counts_grand_and_empty(fraction: float) -> None:
    """The boundary coalitions come first and count toward the budget."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=fraction,
            include_minimal_masks=False,
            seed=1,
        )
    )
    assert len(masks) == int(16 * fraction)
    assert masks[0].present == (True,) * 4
    assert masks[1].present == (False,) * 4


def test_mc_full_budget_terminates_on_full_table() -> None:
    """Full MC budget finishes with every coalition, including empty and grand."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=False,
            seed=0,
        )
    )
    assert len(masks) == 2**4
    assert any(all(mask.present) for mask in masks)
    assert any(not any(mask.present) for mask in masks)


def test_mc_shapley_differs_from_banzhaf_same_seed() -> None:
    """Monte Carlo attributions depend on the selected estimand plugin."""
    player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
    estimator = MonteCarloEstimator()
    masks = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=True,
            seed=11,
        )
    )
    payoffs = _majority_payoffs(tuple(mask.present for mask in masks))
    shapley = estimator.estimate_attributions(
        masks,
        payoffs,
        ShapleyAggregator(),
    )
    banzhaf = estimator.estimate_attributions(
        masks,
        payoffs,
        BanzhafAggregator(),
    )
    assert shapley != banzhaf


def test_mc_reproducible_with_seed() -> None:
    """Identical seeds produce identical mask samples."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    kwargs = {
        "player_set": player_set,
        "budget_fraction": 0.4,
        "include_minimal_masks": False,
        "seed": 99,
    }
    first = [mask.present for mask in MonteCarloEstimator().sample_masks(**kwargs)]
    second = [mask.present for mask in MonteCarloEstimator().sample_masks(**kwargs)]
    assert first == second


def test_mc_converges_toward_exact_on_additive_game() -> None:
    """Large MC budgets approach exact Shapley values on additive games."""
    num_players = 3
    coefficients = (1.0, 2.0, 3.0)
    player_set = PlayerSet(
        player_ids=tuple(f"p{index}" for index in range(num_players))
    )
    all_masks = tuple(
        CoalitionMask.from_sequence(mask)
        for mask in product([False, True], repeat=num_players)
    )
    all_payoffs = [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in all_masks
    ]
    exact = ExactEstimator().estimate_attributions(
        all_masks,
        all_payoffs,
        ShapleyAggregator(),
    )
    mc_masks = list(
        MonteCarloEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=True,
            seed=3,
        )
    )
    mc_payoffs = [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in mc_masks
    ]
    estimated = MonteCarloEstimator().estimate_attributions(
        mc_masks,
        mc_payoffs,
        ShapleyAggregator(),
    )
    assert estimated == pytest.approx(exact, rel=1e-6, abs=1e-6)


def test_mc_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert MonteCarloEstimator().name == "mc"


def test_mc_warns_when_a_player_has_no_observed_marginal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At a budget of two masks only N and the empty set are sampled."""
    players = PlayerSet(player_ids=("a", "b", "c", "d"))
    estimator = MonteCarloEstimator()
    masks = list(estimator.sample_masks(players, 0.1, False, seed=0))
    assert [mask.present for mask in masks] == [(True,) * 4, (False,) * 4]

    with caplog.at_level(logging.WARNING, logger="nlp_shap"):
        values = estimator.estimate_attributions(masks, [4.0, 0.0], ShapleyAggregator())

    assert values == [0.0] * 4
    assert "4 of 4 players have no observed marginal" in caplog.text


def test_mc_does_not_warn_when_every_player_is_observed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    players = PlayerSet(player_ids=("a", "b", "c"))
    estimator = MonteCarloEstimator()
    masks = list(estimator.sample_masks(players, 0.1, True, seed=0))

    with caplog.at_level(logging.WARNING, logger="nlp_shap"):
        estimator.estimate_attributions(
            masks, [float(sum(m.present)) for m in masks], ShapleyAggregator()
        )

    assert "no observed marginal" not in caplog.text
