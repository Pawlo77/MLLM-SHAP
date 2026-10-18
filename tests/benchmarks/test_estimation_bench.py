"""Benchmarks for estimator aggregation on the coalition table.

@see [[src/mllm_shap/estimation/estimands/_weights.py]]
"""

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.complementary import ComplementaryEstimator
from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from mllm_shap.estimation.exact import ExactEstimator
from mllm_shap.estimation.kernel_banzhaf import KernelBanzhafEstimator
from mllm_shap.estimation.leverage import LeverageEstimator
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator
from mllm_shap.estimation.proxy import ProxyEstimator

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
    assert values == pytest.approx(_WEIGHTS.tolist(), abs=0.05)


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


@pytest.mark.bench
def test_leverage_regression_n100_m3200(benchmark: BenchmarkFixture) -> None:
    """Constrained weighted regression on 3 200 sampled coalitions of 100 players."""
    players = PlayerSet(player_ids=tuple(f"p{index}" for index in range(100)))
    weights = np.linspace(0.5, 1.6, 100)
    estimator = LeverageEstimator()
    masks = list(estimator.sample_masks(players, 3200 / 2**100, False, seed=0))
    payoffs = [float(weights[np.asarray(mask.present)].sum()) for mask in masks]

    values = benchmark(
        estimator.estimate_attributions, masks, payoffs, ShapleyAggregator()
    )

    assert len(masks) == 3200
    assert values == pytest.approx(weights.tolist(), abs=1e-6)


@pytest.mark.bench
def test_kernel_banzhaf_regression_n100_m3200(benchmark: BenchmarkFixture) -> None:
    """Centred least squares on 3 200 uniform pairs of 100 players."""
    players = PlayerSet(player_ids=tuple(f"p{index}" for index in range(100)))
    weights = np.linspace(0.5, 1.6, 100)
    estimator = KernelBanzhafEstimator()
    masks = list(estimator.sample_masks(players, 3200 / 2**100, False, seed=0))
    payoffs = [float(weights[np.asarray(mask.present)].sum()) for mask in masks]

    values = benchmark(
        estimator.estimate_attributions, masks, payoffs, BanzhafAggregator()
    )

    assert values == pytest.approx(weights.tolist(), abs=1e-6)


@pytest.mark.bench
def test_proxy_fit_and_values_n12(benchmark: BenchmarkFixture) -> None:
    """Tree proxy, leaf extraction and residual correction on 192 coalitions."""
    estimator = ProxyEstimator()
    masks = list(estimator.sample_masks(_PLAYERS, 192 / 4096, False, seed=0))
    payoffs = _payoffs(masks)

    values = benchmark(
        estimator.estimate_attributions, masks, payoffs, ShapleyAggregator()
    )

    assert values == pytest.approx(_WEIGHTS.tolist(), abs=0.3)
