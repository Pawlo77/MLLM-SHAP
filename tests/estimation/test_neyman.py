"""Tests for Neyman complementary allocation.

@see [[src/mllm_shap/estimation/neyman.py]]
"""

from collections.abc import Callable

import numpy as np
import pytest

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation._shared import compute_complementary_num_splits
from mllm_shap.estimation.neyman import NeymanEstimator


def _additive_payoffs(
    masks: list[CoalitionMask],
    coefficients: tuple[float, ...],
) -> list[float]:
    return [
        sum(
            coefficient
            for coefficient, present in zip(coefficients, mask.present, strict=True)
            if present
        )
        for mask in masks
    ]


def test_neyman_initial_sampling_is_reproducible() -> None:
    """Phase-one Neyman masks are reproducible at a fixed seed."""
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    kwargs = {
        "player_set": player_set,
        "budget_fraction": 0.8,
        "include_minimal_masks": False,
        "seed": 21,
    }
    first = [mask.present for mask in NeymanEstimator().sample_masks(**kwargs)]
    second = [mask.present for mask in NeymanEstimator().sample_masks(**kwargs)]
    assert first == second
    assert len(first) % 2 == 0


def test_neyman_initial_sampling_terminates_for_small_n() -> None:
    """Phase-one sampling must finish for small n (no empty/grand M-cell hang)."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    masks = list(
        NeymanEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=False,
            seed=0,
        )
    )
    assert masks
    assert len(masks) % 2 == 0
    assert len(masks) <= 14


def test_neyman_two_phase_sampling_respects_budget() -> None:
    """Initial and allocation phases together stay within the total budget."""
    player_set = PlayerSet(player_ids=("a", "b", "c", "d"))
    estimator = NeymanEstimator(initial_fraction=0.25, use_standard_method=True)
    phase_one = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=0.6,
            include_minimal_masks=False,
            seed=5,
        )
    )
    payoffs = _additive_payoffs(phase_one, (1.0, 2.0, 3.0, 4.0))
    estimator.begin_allocation(phase_one, payoffs)
    phase_two = list(estimator.sample_allocation_masks())
    assert len(phase_one) + len(phase_two) <= 14


def test_neyman_rejects_minimal_masks_flag() -> None:
    """Neyman sampling rejects include_minimal_masks=True."""
    player_set = PlayerSet(player_ids=("a", "b"))
    with pytest.raises(ValueError, match="include_minimal_masks"):
        _ = list(
            NeymanEstimator().sample_masks(
                player_set,
                budget_fraction=0.5,
                include_minimal_masks=True,
                seed=0,
            )
        )


def test_neyman_allocation_requires_begin_allocation() -> None:
    """Allocation masks cannot be sampled before begin_allocation."""
    estimator = NeymanEstimator()
    player_set = PlayerSet(player_ids=("a", "b", "c"))
    _ = list(
        estimator.sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=False,
            seed=1,
        )
    )
    with pytest.raises(RuntimeError, match="allocation has not been initialized"):
        _ = list(estimator.sample_allocation_masks())


def test_neyman_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert NeymanEstimator().name == "neyman_cc"


def test_neyman_phase_one_fills_feasible_grid_and_uses_budget() -> None:
    """Phase one ends when every feasible cell is filled; phase two spends the rest."""
    num_players = 6
    player_set = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    estimator = NeymanEstimator()
    phase_one = list(estimator.sample_masks(player_set, 1.0, False, seed=0))
    payoffs = [sum(m.present) / num_players for m in phase_one]
    estimator.begin_allocation(phase_one, payoffs)
    phase_two = list(estimator.sample_allocation_masks())
    # Dedup may drop a few phase-two draws; the budget must still be mostly spent.
    assert len(phase_one) + len(phase_two) >= int(0.75 * 2**num_players)
    assert len(phase_two) > 0


def test_neyman_complement_column_pairs_size_j_with_n_minus_j() -> None:
    assert NeymanEstimator._complement_column(2, num_players=5) == 3
    assert NeymanEstimator._complement_column(4, num_players=5) == 1


def _two_phase(
    num_players: int,
    fraction: float,
    payoff: Callable[[tuple[bool, ...]], float],
    seed: int = 0,
) -> tuple[list[CoalitionMask], list[CoalitionMask], np.ndarray]:
    """Run both phases; return phase-one masks, phase-two masks and allocation."""
    player_set = PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))
    estimator = NeymanEstimator()
    phase_one = list(estimator.sample_masks(player_set, fraction, False, seed=seed))
    estimator.begin_allocation(phase_one, [payoff(m.present) for m in phase_one])
    assert estimator._m_hat is not None
    allocation = estimator._m_hat.copy()
    phase_two = list(estimator.sample_allocation_masks())
    return phase_one, phase_two, allocation


@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.8])
def test_neyman_never_exceeds_budget(fraction: float) -> None:
    num_players = 7
    phase_one, phase_two, _ = _two_phase(
        num_players, fraction, lambda present: float(sum(present) ** 2)
    )
    budget = compute_complementary_num_splits(num_players, fraction, False)
    assert len(phase_one) + len(phase_two) <= budget
    assert len({m.present for m in phase_one + phase_two}) == len(phase_one) + len(
        phase_two
    )


def test_neyman_allocates_more_to_high_variance_stratum() -> None:
    """Noise only in coalitions of size 3 draws the allocation to that size."""
    rng = np.random.default_rng(3)
    noise = {}

    def payoff(present: tuple[bool, ...]) -> float:
        if sum(present) in (3, 5):
            return noise.setdefault(present, float(rng.normal(0.0, 10.0)))
        return float(sum(present))

    _, _, allocation = _two_phase(8, 0.5, payoff)
    assert allocation[5] > allocation[6]
    assert allocation[5] > allocation[7]


def test_neyman_zero_variance_falls_back_to_capacity() -> None:
    """A constant game still spends phase two, split by unseen pairs per size."""
    _, phase_two, allocation = _two_phase(6, 1.0, lambda present: 1.0)
    assert allocation.sum() > 0
    assert len(phase_two) == 2 * allocation.sum()


def test_neyman_truncated_phase_one_spreads_empty_cells() -> None:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(8)))
    for seed in range(10):
        estimator = NeymanEstimator()
        list(estimator.sample_masks(players, 0.1, False, seed))
        assert estimator.m_counts is not None
        empty = (estimator.m_counts[:, 1:8] == 0).sum(1)
        assert empty.max() - empty.min() <= 1


def test_neyman_phase_one_seeds_every_size_one_stratum() -> None:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(8)))
    estimator = NeymanEstimator()
    masks = [m.present for m in estimator.sample_masks(players, 0.1, False, 0)]
    singletons = [tuple(i == p for i in range(8)) for p in range(8)]
    assert masks[2:18] == [
        mask for single in singletons for mask in (single, tuple(not v for v in single))
    ]
    assert estimator.m_counts is not None
    assert estimator.m_counts[:, 1].min() > 0
