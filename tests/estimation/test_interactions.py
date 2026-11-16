"""Pairwise interactions through shapiq against the enumerated reference games.

@see [[src/mllm_shap/estimation/interactions.py]]
"""

import numpy as np
import pytest

pytest.importorskip("shapiq")

from mllm_shap.estimation.interactions import pairwise_interactions
from tests.estimation._games import Game, reference_games


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_exact_pairwise_values_equal_the_reference_games(game: Game) -> None:
    masks, payoffs = game.table()
    matrix = pairwise_interactions(masks, payoffs)
    assert matrix == pytest.approx(game.interactions, abs=1e-8)


def test_the_index_can_be_chosen() -> None:
    game = reference_games(6)[1]
    masks, payoffs = game.table()
    sii = pairwise_interactions(masks, payoffs, index="SII")
    k_sii = pairwise_interactions(masks, payoffs, index="k-SII")
    assert sii == pytest.approx(k_sii, abs=1e-8)
    assert np.allclose(sii, sii.T)


def test_a_partial_table_is_refused() -> None:
    game = reference_games(5)[0]
    masks, payoffs = game.table()
    with pytest.raises(ValueError, match="complete table"):
        pairwise_interactions(masks[:-1], payoffs[:-1])


def test_a_missing_shapiq_install_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "shapiq", None)
    game = reference_games(4)[0]
    masks, payoffs = game.table()
    with pytest.raises(ImportError, match=r"mllm-shap\[interactions\]"):
        pairwise_interactions(masks, payoffs)
