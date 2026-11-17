"""Beta Shapley and weighted Banzhaf against the enumerated reference games.

@see [[tests/estimation/_games.py]]
"""

import numpy as np
import pytest
from pydantic import ValidationError

from mllm_shap.domain.estimands import Estimand, estimand_to_wire
from mllm_shap.domain.players import PlayerSet
from mllm_shap.errors import ConfigError
from mllm_shap.estimation.estimands import (
    BetaShapleyAggregator,
    ShapleyAggregator,
    WeightedBanzhafAggregator,
)
from mllm_shap.estimation.estimands.semivalue import (
    beta_shapley_weight,
    weighted_banzhaf_weight,
)
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator
from mllm_shap.pipeline.config import (
    BackendConfig,
    ExplainConfig,
    explain_config_from_yaml,
)
from mllm_shap.plugins.context import PluginContext
from tests.estimation._games import (
    BETA_SHAPLEY,
    WEIGHTED_BANZHAF_P,
    Game,
    reference_games,
)


def _context(options: dict[str, object]) -> PluginContext:
    config = ExplainConfig(backend=BackendConfig(kind="mock", model_id="m"))
    return PluginContext(config=config, options=options)


def _beta() -> BetaShapleyAggregator:
    alpha, beta = BETA_SHAPLEY
    return BetaShapleyAggregator(_context({"alpha": alpha, "beta": beta}))


def _weighted() -> WeightedBanzhafAggregator:
    return WeightedBanzhafAggregator(_context({"p": WEIGHTED_BANZHAF_P}))


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_full_tables_give_the_exact_semivalues(game: Game) -> None:
    masks, payoffs = game.table()
    assert _beta().aggregate(masks, payoffs) == pytest.approx(game.beta_shapley)
    assert _weighted().aggregate(masks, payoffs) == pytest.approx(game.weighted_banzhaf)


@pytest.mark.parametrize("num_players", [1, 2, 5, 9])
def test_weights_sum_to_one_over_all_subsets(num_players: int) -> None:
    from math import comb

    for weight in (
        lambda s: beta_shapley_weight(s, num_players, 4.0, 1.0),
        lambda s: weighted_banzhaf_weight(s, num_players, 0.3),
    ):
        total = sum(comb(num_players - 1, s) * weight(s) for s in range(num_players))
        assert total == pytest.approx(1.0)


def test_unit_parameters_recover_shapley() -> None:
    game = reference_games(6)[3]
    masks, payoffs = game.table()
    beta = BetaShapleyAggregator(_context({"alpha": 1.0, "beta": 1.0}))
    assert beta.aggregate(masks, payoffs) == pytest.approx(
        ShapleyAggregator().aggregate(masks, payoffs)
    )


def test_one_half_recovers_banzhaf() -> None:
    game = reference_games(6)[3]
    masks, payoffs = game.table()
    weighted = WeightedBanzhafAggregator(_context({"p": 0.5}))
    assert weighted.aggregate(masks, payoffs) == pytest.approx(game.banzhaf)


@pytest.mark.parametrize("name", ["beta", "weighted"])
def test_mc_samples_converge_to_the_exact_values(name: str) -> None:
    game = reference_games(8)[1]
    aggregator = _beta() if name == "beta" else _weighted()
    exact = game.beta_shapley if name == "beta" else game.weighted_banzhaf
    errors = []
    for seed in range(20):
        masks = list(
            MonteCarloEstimator().sample_masks(
                PlayerSet(player_ids=tuple(map(str, range(8)))), 0.9, False, seed
            )
        )
        payoffs = [game.payoff(mask.present) for mask in masks]
        errors.append(
            np.abs(
                np.array(aggregator.aggregate([m.present for m in masks], payoffs))
                - exact
            )
        )
    assert np.mean(errors) < 0.05


def test_wire_labels_and_options_are_checked() -> None:
    assert estimand_to_wire(Estimand.BETA_SHAPLEY) == "beta_shapley"
    assert estimand_to_wire(Estimand.WEIGHTED_BANZHAF) == "weighted_banzhaf"
    with pytest.raises(ValidationError):
        WeightedBanzhafAggregator(_context({"p": 1.0}))
    with pytest.raises(ValidationError):
        BetaShapleyAggregator(_context({"alpha": 0.0}))


@pytest.mark.parametrize("estimator", ["neyman_cc", "complementary", "leverage"])
def test_config_refuses_estimators_that_sample_in_pairs(estimator: str) -> None:
    head = "backend: {kind: mock, model_id: stub}\n"
    with pytest.raises(ConfigError, match="'mc' or 'exact'"):
        explain_config_from_yaml(
            f"{head}explanation: {{estimator: {estimator}, estimand: beta_shapley}}\n"
        )
    config = explain_config_from_yaml(
        f"{head}explanation: {{estimator: mc, estimand: weighted_banzhaf}}\n"
    )
    assert config.explanation.estimand is Estimand.WEIGHTED_BANZHAF


def test_the_aggregators_name_their_estimand_and_weigh_coalitions() -> None:
    assert _beta().estimand is Estimand.BETA_SHAPLEY
    assert _weighted().estimand is Estimand.WEIGHTED_BANZHAF
    assert _beta().coalition_weight(2, 6) == pytest.approx(
        beta_shapley_weight(2, 6, *BETA_SHAPLEY)
    )
    assert _weighted().coalition_weight(2, 6) == pytest.approx(
        weighted_banzhaf_weight(2, 6, WEIGHTED_BANZHAF_P)
    )
