"""Kernel Banzhaf: exactness, noise robustness and accuracy against sample reuse.

@see [[src/mllm_shap/estimation/kernel_banzhaf.py]]
"""

import numpy as np
import pytest
from pydantic import ValidationError

from mllm_shap import ExplainConfig
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation._shared import budget_fraction
from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from mllm_shap.estimation.kernel_banzhaf import KernelBanzhafEstimator
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator
from tests.estimation._games import Game, reference_games


def _error(
    game: Game, estimator_type: type, multiple: int, noise: float, seeds: int
) -> float:
    num_players = game.num_players
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    fraction = budget_fraction(num_players, None, multiple * num_players)
    errors = []
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        estimator = estimator_type()
        masks = list(estimator.sample_masks(players, fraction, False, seed))
        payoffs = [game.payoff(m.present) + noise * rng.normal() for m in masks]
        values = estimator.estimate_attributions(masks, payoffs, BanzhafAggregator())
        errors.append(np.linalg.norm(np.array(values) - game.banzhaf))
    return float(np.mean(errors))


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_exact_at_full_budget(game: Game) -> None:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(8)))
    estimator = KernelBanzhafEstimator()
    masks = list(estimator.sample_masks(players, 1.0, False, 0))
    values = estimator.estimate_attributions(
        masks, [game.payoff(m.present) for m in masks], BanzhafAggregator()
    )
    assert len(masks) == 2**8
    assert values == pytest.approx(game.banzhaf, abs=1e-9)


def test_sampler_pairs_are_distinct_complements() -> None:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(10)))
    masks = [
        m.present
        for m in KernelBanzhafEstimator().sample_masks(
            players, budget_fraction(10, None, 50), False, 2
        )
    ]
    assert masks[:2] == [(True,) * 10, (False,) * 10]
    assert len(masks) == len(set(masks)) == 50
    for left, right in zip(masks[2::2], masks[3::2], strict=True):
        assert right == tuple(not value for value in left)


def test_rejects_the_shapley_estimand() -> None:
    estimator = KernelBanzhafEstimator()
    with pytest.raises(ValueError, match="banzhaf estimand only"):
        estimator.estimate_attributions([], [], ShapleyAggregator())
    with pytest.raises(ValidationError, match="kernel_banzhaf"):
        ExplainConfig.model_validate({
            "backend": {"kind": "mock", "model_id": "mock"},
            "explanation": {"estimator": "kernel_banzhaf"},
        })


@pytest.mark.parametrize("noise", [0.0, 0.3])
@pytest.mark.parametrize("multiple", [8, 16, 32])
def test_beats_sample_reuse_monte_carlo(multiple: int, noise: float) -> None:
    for game in reference_games(10)[1:]:
        regression = _error(game, KernelBanzhafEstimator, multiple, noise, 25)
        reuse = _error(game, MonteCarloEstimator, multiple, noise, 25)
        assert regression < reuse, game.name
