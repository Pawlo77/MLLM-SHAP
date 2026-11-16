"""LIME surrogate against scikit-learn's weighted ridge and the sampler's contract.

@see [[src/mllm_shap/estimation/lime.py]]
"""

import numpy as np
import pytest

from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.estimands import ShapleyAggregator
from mllm_shap.estimation.lime import LimeEstimator, lime_values
from tests.estimation._games import additive_game


def _sample(n: int = 8, fraction: float = 0.4, seed: int = 0):
    players = PlayerSet(player_ids=tuple(map(str, range(n))))
    return list(LimeEstimator().sample_masks(players, fraction, False, seed))


def test_coefficients_match_scikit_learn_ridge_with_the_lime_kernel() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(1)
    present = rng.integers(0, 2, (60, 9)).astype(float)
    present[0] = 1.0
    payoffs = rng.normal(size=60)
    distance = 100.0 * (1.0 - np.sqrt(present.sum(axis=1) / 9))
    weights = np.sqrt(np.exp(-(distance**2) / 25.0**2))
    ridge = linear_model.Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(present, payoffs, sample_weight=weights)
    assert lime_values(present, payoffs) == pytest.approx(ridge.coef_, abs=1e-6)


def test_additive_game_recovers_its_weights_up_to_the_ridge_shrinkage() -> None:
    game = additive_game(8)
    masks = _sample(8, 0.9)
    payoffs = [game.payoff(mask.present) for mask in masks]
    values = LimeEstimator().estimate_attributions(masks, payoffs, ShapleyAggregator())
    assert np.corrcoef(values, game.shapley)[0, 1] > 0.99


def test_masks_are_distinct_within_budget_and_seeded() -> None:
    masks = _sample(8, 0.3, seed=4)
    assert len({m.present for m in masks}) == len(masks) <= int(256 * 0.3)
    assert masks[0].present == (True,) * 8
    assert [m.present for m in masks] == [m.present for m in _sample(8, 0.3, seed=4)]


def test_estimand_is_ignored() -> None:
    masks = _sample()
    payoffs = list(np.random.default_rng(0).normal(size=len(masks)))
    estimator = LimeEstimator()
    assert estimator.estimate_attributions(
        masks, payoffs, None
    ) == estimator.estimate_attributions(masks, payoffs, ShapleyAggregator())
