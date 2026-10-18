"""Tests for Monte Carlo coalition sampling.

@see [[src/mllm_shap/estimation/monte_carlo.py]]
"""

import logging
from itertools import product

import numpy as np
import pytest

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation._shared import budget_fraction
from mllm_shap.estimation.estimands.banzhaf import BanzhafAggregator
from mllm_shap.estimation.estimands.shapley import ShapleyAggregator
from mllm_shap.estimation.exact import ExactEstimator
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator
from tests.estimation._games import Game, reference_games


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

    with caplog.at_level(logging.WARNING, logger="mllm_shap"):
        values = estimator.estimate_attributions(masks, [4.0, 0.0], ShapleyAggregator())

    assert values == [0.0] * 4
    assert "4 of 4 players have no stratum sampled" in caplog.text


def test_mc_does_not_warn_when_every_player_is_observed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    players = PlayerSet(player_ids=("a", "b", "c"))
    estimator = MonteCarloEstimator()
    masks = list(estimator.sample_masks(players, 0.1, True, seed=0))

    with caplog.at_level(logging.WARNING, logger="mllm_shap"):
        estimator.estimate_attributions(
            masks, [float(sum(m.present)) for m in masks], ShapleyAggregator()
        )

    assert "have no stratum sampled" not in caplog.text


def _pair_reference(
    masks: list[CoalitionMask], payoffs: list[float], num_players: int
) -> list[float]:
    """Previous aggregation: stratified mean over sampled pairs ``(S, S + {i})``."""
    lookup = {mask.present: payoff for mask, payoff in zip(masks, payoffs, strict=True)}
    values = []
    for player in range(num_players):
        strata: dict[int, list[float]] = {}
        for present, payoff in lookup.items():
            if present[player]:
                continue
            extended = (*present[:player], True, *present[player + 1 :])
            if extended in lookup:
                strata.setdefault(sum(present), []).append(lookup[extended] - payoff)
        means = [sum(v) / len(v) for v in strata.values()]
        values.append(sum(means) / len(means) if means else 0.0)
    return values


def _uniform_masks(num_players: int, count: int, seed: int) -> list[CoalitionMask]:
    """Previous sampler: grand, empty, then distinct uniform coalitions."""
    rng = np.random.default_rng(seed)
    seen = {(True,) * num_players, (False,) * num_players}
    while len(seen) < count:
        seen.add(tuple(bool(bit) for bit in rng.integers(0, 2, num_players)))
    return [CoalitionMask.from_sequence(present) for present in seen]


def _mc_estimates(game: Game, multiple: int, seeds: int) -> list[list[float]]:
    num_players = game.num_players
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    fraction = budget_fraction(num_players, None, multiple * num_players)
    estimates = []
    for seed in range(seeds):
        estimator = MonteCarloEstimator()
        masks = list(estimator.sample_masks(players, fraction, False, seed))
        payoffs = [game.payoff(mask.present) for mask in masks]
        estimates.append(
            estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
        )
    return estimates


def test_mc_sampler_order_sizes_and_budget() -> None:
    """Boundary, singleton and co-singleton masks come first; no repeats."""
    num_players = 8
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    masks = [
        mask.present
        for mask in MonteCarloEstimator().sample_masks(
            players, budget_fraction(num_players, None, 60), False, seed=5
        )
    ]
    assert len(masks) == len(set(masks)) == 60
    assert [sum(mask) for mask in masks[: 2 + 2 * num_players]] == [
        8,
        0,
        *[1] * num_players,
        *[7] * num_players,
    ]
    assert {sum(mask) for mask in masks[2 + 2 * num_players :]} <= set(range(2, 7))


@pytest.mark.parametrize("game", reference_games(10)[:5], ids=lambda game: game.name)
def test_mc_is_nearly_unbiased_and_beats_pair_reuse(game: Game) -> None:
    estimates = np.array(_mc_estimates(game, 32, 200))
    bias = np.abs(estimates.mean(axis=0) - game.shapley)
    standard_error = estimates.std(axis=0) / np.sqrt(len(estimates))
    assert (bias <= 4 * standard_error + 0.02 * np.linalg.norm(game.shapley)).all()

    if game.name == "additive":
        return  # pair differences are constant there, so pair reuse is exact
    for multiple in (4, 32):
        new, old = [], []
        for seed, value in enumerate(_mc_estimates(game, multiple, 60)):
            masks = _uniform_masks(10, multiple * 10, seed)
            payoffs = [game.payoff(mask.present) for mask in masks]
            new.append(np.linalg.norm(np.array(value) - game.shapley))
            old.append(
                np.linalg.norm(
                    np.array(_pair_reference(masks, payoffs, 10)) - game.shapley
                )
            )
        assert np.mean(new) < np.mean(old)
