"""Permutation sampling: budget, coverage of prefixes and convergence.

@see [[src/mllm_shap/estimation/permutation.py]]
"""

import numpy as np
import pytest

from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from mllm_shap.estimation.permutation import PermutationEstimator
from tests.estimation._games import Game, reference_games


def _players(n: int) -> PlayerSet:
    return PlayerSet(player_ids=tuple(map(str, range(n))))


def _estimator(antithetic: bool = True) -> PermutationEstimator:
    estimator = PermutationEstimator()
    estimator.antithetic = antithetic
    return estimator


def _mean_squared_error(game: Game, antithetic: bool, seeds: int = 60) -> float:
    errors = []
    for seed in range(seeds):
        masks = list(
            _estimator(antithetic).sample_masks(_players(8), 0.35, False, seed)
        )
        payoffs = [game.payoff(mask.present) for mask in masks]
        values = ShapleyAggregator().aggregate([m.present for m in masks], payoffs)
        errors.append(np.mean((np.array(values) - game.shapley) ** 2))
    return float(np.mean(errors))


@pytest.mark.parametrize("fraction", [0.1, 0.5, 1.0])
def test_budget_is_honoured_and_masks_are_unique(fraction: float) -> None:
    masks = list(_estimator().sample_masks(_players(8), fraction, False, 3))
    assert len(masks) <= max(2, int(256 * fraction))
    assert len({mask.present for mask in masks}) == len(masks)
    assert masks[0].present == (True,) * 8
    assert masks[1].present == (False,) * 8


def test_antithetic_orderings_include_the_complement_chain() -> None:
    # 12 of 64 coalitions: the grand and empty ones and one ordering with its reverse.
    masks = {
        m.present for m in _estimator().sample_masks(_players(6), 12 / 64, False, 1)
    }
    assert len(masks) == 12
    assert {tuple(not v for v in present) for present in masks} == masks


def test_same_seed_gives_the_same_masks() -> None:
    first = [m.present for m in _estimator().sample_masks(_players(7), 0.3, False, 5)]
    again = [m.present for m in _estimator().sample_masks(_players(7), 0.3, False, 5)]
    assert first == again


@pytest.mark.parametrize("game", reference_games(8)[:2], ids=lambda game: game.name)
def test_values_converge_to_the_exact_shapley_values(game: Game) -> None:
    masks = list(_estimator().sample_masks(_players(8), 0.9, False, 0))
    payoffs = [game.payoff(mask.present) for mask in masks]
    values = PermutationEstimator().estimate_attributions(
        masks, payoffs, ShapleyAggregator()
    )
    assert values == pytest.approx(game.shapley, abs=0.15)


def test_banzhaf_estimand_is_served_from_the_same_prefixes() -> None:
    game = reference_games(8)[1]
    masks = list(_estimator().sample_masks(_players(8), 0.9, False, 0))
    payoffs = [game.payoff(mask.present) for mask in masks]
    values = BanzhafAggregator().aggregate([m.present for m in masks], payoffs)
    assert values == pytest.approx(game.banzhaf, abs=0.15)


def test_antithetic_orderings_do_not_increase_the_error() -> None:
    game = reference_games(8)[1]
    assert _mean_squared_error(game, True) <= 1.05 * _mean_squared_error(game, False)
