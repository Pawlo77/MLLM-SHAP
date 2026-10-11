"""Benchmarks for estimator aggregation on the coalition table.

@see [[src/mllm_shap/estimation/estimands/_weights.py]]
"""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.complementary import ComplementaryEstimator
from mllm_shap.estimation.estimands import ShapleyAggregator
from mllm_shap.estimation.exact import ExactEstimator
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator

_NUM_PLAYERS = 12
_PLAYERS = PlayerSet(player_ids=tuple(f"p{index}" for index in range(_NUM_PLAYERS)))
_WEIGHTS = np.linspace(0.5, 1.6, _NUM_PLAYERS)


def _payoffs(masks: list[CoalitionMask]) -> list[float]:
    """Additive game, so every estimator should return ``_WEIGHTS``."""
    return [float(_WEIGHTS[np.asarray(mask.present)].sum()) for mask in masks]


@pytest.mark.bench
def test_exact_aggregation_full_table_n12(benchmark: BenchmarkFixture) -> None:
    """Exact Shapley over all 4 096 coalitions of 12 players."""
    estimator = ExactEstimator()
    masks = list(estimator.sample_masks(_PLAYERS, 1.0, False, seed=0))
    payoffs = _payoffs(masks)

    values = benchmark(
        estimator.estimate_attributions, masks, payoffs, ShapleyAggregator()
    )

    assert values == pytest.approx(_WEIGHTS.tolist())


@pytest.mark.bench
def test_stratified_marginals_m2000(benchmark: BenchmarkFixture) -> None:
    """Stratified marginal aggregation of about 2 000 sampled coalitions."""
    estimator = MonteCarloEstimator()
    masks = list(estimator.sample_masks(_PLAYERS, 2000 / 4096, False, seed=0))
    payoffs = _payoffs(masks)

    values = benchmark(
        estimator.estimate_attributions, masks, payoffs, ShapleyAggregator()
    )

    assert 1900 <= len(masks) <= 2100
    assert values == pytest.approx(_WEIGHTS.tolist())


@pytest.mark.bench
def test_complementary_aggregation_n12(benchmark: BenchmarkFixture) -> None:
    """Complementary-contribution aggregation of half the coalition table."""
    estimator = ComplementaryEstimator()
    masks = list(estimator.sample_masks(_PLAYERS, 0.5, False, seed=0))
    payoffs = _payoffs(masks)

    values = benchmark(
        estimator.estimate_attributions, masks, payoffs, ShapleyAggregator()
    )

    # Partial-budget CC is not exactly efficient, but with every size-1 stratum
    # seeded the sum stays close to v(N) - v(empty) (it was 16.99 against 12.6
    # before the fix; see [[Fixed issues|C12]]).
    assert len(values) == _NUM_PLAYERS
    assert sum(values) == pytest.approx(_WEIGHTS.sum(), abs=0.25)
    assert values == pytest.approx(_WEIGHTS.tolist(), abs=0.15)
