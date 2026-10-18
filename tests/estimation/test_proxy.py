"""ProxySHAP: exact tree values, consistency, determinism and the missing extra.

@see [[src/mllm_shap/estimation/proxy.py]]
"""

import sys

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from mllm_shap.domain.estimands import Estimand
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation._shared import budget_fraction
from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from mllm_shap.estimation.estimands._weights import aggregate_shapley_exact
from mllm_shap.estimation.leverage import LeverageEstimator
from mllm_shap.estimation.proxy import ProxyEstimator, tree_values
from tests.estimation._games import Game, reference_games


@pytest.mark.parametrize("estimand", [Estimand.SHAPLEY, Estimand.BANZHAF])
def test_tree_values_are_the_exact_values_of_the_ensemble(estimand: Estimand) -> None:
    game = next(g for g in reference_games(7) if g.name == "pairwise")
    masks, payoffs = game.table()
    present = np.array(masks, dtype=float)
    model = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0).fit(
        present, payoffs
    )
    predictions = model.predict(present).tolist()
    if estimand is Estimand.SHAPLEY:
        expected = aggregate_shapley_exact(masks, predictions)
    else:
        expected = BanzhafAggregator().aggregate(masks, predictions)
    assert tree_values(model, 7, estimand) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("game", reference_games(8)[:5], ids=lambda game: game.name)
def test_exact_at_full_budget_for_both_estimands(game: Game) -> None:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(8)))
    estimator = ProxyEstimator()
    masks = list(estimator.sample_masks(players, 1.0, False, 0))
    payoffs = [game.payoff(m.present) for m in masks]
    for aggregator, exact in (
        (ShapleyAggregator(), game.shapley),
        (BanzhafAggregator(), game.banzhaf),
    ):
        values = estimator.estimate_attributions(masks, payoffs, aggregator)
        assert values == pytest.approx(exact, abs=1e-8)


def test_values_are_deterministic_for_reanalysis() -> None:
    game = reference_games(10)[3]
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(10)))
    fraction = budget_fraction(10, None, 160)
    masks = list(ProxyEstimator().sample_masks(players, fraction, False, 4))
    payoffs = [game.payoff(m.present) for m in masks]
    first = ProxyEstimator().estimate_attributions(masks, payoffs, ShapleyAggregator())
    again = ProxyEstimator().estimate_attributions(masks, payoffs, ShapleyAggregator())
    assert first == again


@pytest.mark.parametrize("name", ["airport", "voting", "unanimity", "mock_prompt"])
def test_beats_leverage_at_sixteen_times_n(name: str) -> None:
    game = next(g for g in reference_games(10) if g.name == name)
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(10)))
    fraction = budget_fraction(10, None, 160)
    errors = {}
    for estimator_type in (ProxyEstimator, LeverageEstimator):
        total = 0.0
        for seed in range(10):
            estimator = estimator_type()
            masks = list(estimator.sample_masks(players, fraction, False, seed))
            payoffs = [game.payoff(m.present) for m in masks]
            values = estimator.estimate_attributions(
                masks, payoffs, ShapleyAggregator()
            )
            total += np.linalg.norm(np.array(values) - game.shapley)
        errors[estimator_type] = total
    assert errors[ProxyEstimator] < errors[LeverageEstimator]


def test_missing_scikit_learn_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", None)
    with pytest.raises(ImportError, match=r"mllm-shap\[proxy\]"):
        ProxyEstimator()


def test_large_games_with_small_budgets_skip_the_residual_correction() -> None:
    """From 30 players and below 16 masks per player only the proxy values remain."""
    num_players = 32
    weights = np.linspace(0.0, 1.0, num_players)
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    estimator = ProxyEstimator()
    masks = list(estimator.sample_masks(players, 200 / 2**num_players, False, 0))
    payoffs = [float(weights @ np.array(m.present)) for m in masks]
    values = estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
    assert len(values) == num_players
    assert np.isfinite(values).all()
    assert estimator.name == "proxy"
