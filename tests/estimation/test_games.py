"""Reference games: exact values must be efficient and agree with the full table.

@see [[tests/estimation/_games.py]]
"""

import numpy as np
import pytest

from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from tests.estimation._games import (
    Game,
    _game,
    pairwise_game,
    recorded_teacher_forced_game,
    reference_games,
)


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_exact_values_match_the_full_table(game: Game) -> None:
    masks, payoffs = game.table()
    grand, empty = payoffs[-1], payoffs[0]
    assert game.shapley.sum() == pytest.approx(grand - empty)
    assert ShapleyAggregator().aggregate(masks, payoffs) == pytest.approx(game.shapley)
    assert np.allclose(
        BanzhafAggregator().aggregate(masks, payoffs), game.banzhaf, atol=1e-9
    )


def test_every_semivalue_gives_one_per_player_in_the_count_game() -> None:
    game = _game("count", 7, lambda mask: float(sum(mask)))
    for values in (
        game.shapley,
        game.banzhaf,
        game.beta_shapley,
        game.weighted_banzhaf,
    ):
        assert values == pytest.approx(np.ones(7))
    assert not game.interactions.any()


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_interactions_are_symmetric_with_a_zero_diagonal(game: Game) -> None:
    assert np.allclose(game.interactions, game.interactions.T)
    assert not np.diag(game.interactions).any()


def test_pairwise_game_interactions_are_its_pair_weights() -> None:
    game = pairwise_game(8)
    rng = np.random.default_rng(0)
    singles = rng.uniform(-1.0, 1.0, 8)
    pairs = np.triu(rng.uniform(-1.0, 1.0, (8, 8)), 1)
    assert game.interactions == pytest.approx(pairs + pairs.T)
    assert game.shapley == pytest.approx(singles + game.interactions.sum(axis=1) / 2)


def test_the_recorded_teacher_forced_table_is_complete_and_efficient() -> None:
    game = recorded_teacher_forced_game()
    masks, payoffs = game.table()
    assert len(masks) == 1024
    assert game.shapley.sum() == pytest.approx(payoffs[-1] - payoffs[0])
    for values in (game.banzhaf, game.beta_shapley, game.weighted_banzhaf):
        assert np.all(np.isfinite(values))
    assert ShapleyAggregator().aggregate(masks, payoffs) == pytest.approx(game.shapley)
