"""Reference games: exact values must be efficient and agree with the full table.

@see [[tests/estimation/_games.py]]
"""

import numpy as np
import pytest

from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from tests.estimation._games import Game, reference_games


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_exact_values_match_the_full_table(game: Game) -> None:
    masks, payoffs = game.table()
    grand, empty = payoffs[-1], payoffs[0]
    assert game.shapley.sum() == pytest.approx(grand - empty)
    assert ShapleyAggregator().aggregate(masks, payoffs) == pytest.approx(game.shapley)
    assert np.allclose(
        BanzhafAggregator().aggregate(masks, payoffs), game.banzhaf, atol=1e-9
    )
