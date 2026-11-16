"""Standard errors of the stratified estimators: formula, coverage and pipeline option.

@see [[src/mllm_shap/estimation/estimands/_weights.py]]
"""

import numpy as np
import pytest

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet
from mllm_shap.errors import ConfigError
from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from mllm_shap.estimation.estimands._weights import marginal_standard_errors
from mllm_shap.estimation.exact import ExactEstimator
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator
from mllm_shap.pipeline.config import explain_config_from_yaml
from tests.estimation._games import Game, reference_games

PLAYERS = PlayerSet(player_ids=tuple(map(str, range(8))))


def _draw(game: Game, seed: int, fraction: float):
    masks = list(MonteCarloEstimator().sample_masks(PLAYERS, fraction, False, seed))
    return masks, [game.payoff(mask.present) for mask in masks]


def test_a_full_table_has_no_standard_error() -> None:
    game = reference_games(6)[1]
    masks, payoffs = game.table()
    errors = marginal_standard_errors(
        masks, payoffs, ShapleyAggregator().coalition_weight
    )
    assert errors == pytest.approx([0.0] * 6)
    exact = ExactEstimator().standard_errors(
        [CoalitionMask.from_sequence(m) for m in masks], payoffs, ShapleyAggregator()
    )
    assert exact == [0.0] * 6


def test_errors_shrink_as_the_budget_grows() -> None:
    game = reference_games(8)[1]
    small, large = [], []
    for seed in range(20):
        for fraction, sink in ((0.1, small), (0.6, large)):
            masks, payoffs = _draw(game, seed, fraction)
            sink.append(
                np.mean(
                    marginal_standard_errors(
                        [m.present for m in masks],
                        payoffs,
                        ShapleyAggregator().coalition_weight,
                    )
                )
            )
    assert np.mean(large) < np.mean(small)


@pytest.mark.parametrize("aggregator", [ShapleyAggregator(), BanzhafAggregator()])
def test_intervals_cover_the_exact_values_about_95_percent(aggregator) -> None:
    game = reference_games(8)[1]
    exact = game.shapley if isinstance(aggregator, ShapleyAggregator) else game.banzhaf
    hits = total = 0
    for seed in range(150):
        masks, payoffs = _draw(game, seed, 0.25)
        present = [m.present for m in masks]
        values = np.array(aggregator.aggregate(present, payoffs))
        errors = np.array(
            marginal_standard_errors(present, payoffs, aggregator.coalition_weight)
        )
        hits += int(np.sum(np.abs(values - exact) <= 1.96 * errors))
        total += len(values)
    assert 0.90 <= hits / total <= 0.99


def test_config_asks_for_errors_only_where_they_are_defined() -> None:
    head = "backend: {kind: mock, model_id: stub}\n"
    config = explain_config_from_yaml(
        f"{head}explanation: {{estimator: mc, stderr: true}}\n"
    )
    assert config.explanation.stderr is True
    assert "stderr" not in explain_config_from_yaml(head).explanation.model_dump()
    for bad in (
        "{estimator: leverage, stderr: true}",
        "{estimator: mc, stderr: true, efficiency: additive}",
        "{estimator: mc, stderr: true, normalizer: min_max}",
    ):
        with pytest.raises(ConfigError, match="stderr"):
            explain_config_from_yaml(f"{head}explanation: {bad}\n")
