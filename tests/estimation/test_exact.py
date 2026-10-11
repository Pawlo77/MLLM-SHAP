"""Tests for exact coalition enumeration and estimand wiring.

@see [[src/mllm_shap/estimation/exact.py]]
"""

from itertools import product

import numpy as np
import pytest

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.estimands import Estimand
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.estimands.banzhaf import BanzhafAggregator
from mllm_shap.estimation.estimands.shapley import ShapleyAggregator
from mllm_shap.estimation.exact import ExactEstimator


def _reference_shapley_values(
    masks: np.ndarray,
    payoffs: np.ndarray,
) -> np.ndarray:
    """Port of legacy PreciseShapExplainer._calculate_shap_values for parity checks."""
    num_features = masks.shape[1]
    shap_values = np.zeros(num_features, dtype=np.float64)
    indices = np.arange(num_features + 1, dtype=np.float64)
    indices[0] = 1.0
    factorials = np.cumprod(indices)
    power_of_2 = 2 ** np.arange(num_features)
    subset_hashes = (masks.astype(np.float64) * power_of_2).sum(axis=1)
    sorted_hashes = np.sort(subset_hashes)
    sort_idx = np.argsort(subset_hashes)
    sorted_outputs = payoffs[sort_idx]
    subset_sizes = masks.sum(axis=1)

    for player_index in range(num_features):
        include_mask = masks[:, player_index]
        included_outputs = payoffs[include_mask]
        included_hashes = subset_hashes[include_mask]
        excluded_hash = included_hashes - power_of_2[player_index]
        excluded_outputs = sorted_outputs[np.searchsorted(sorted_hashes, excluded_hash)]
        excluded_subset_sizes = subset_sizes[include_mask] - 1
        weights = (
            factorials[excluded_subset_sizes]
            * factorials[num_features - excluded_subset_sizes - 1]
            / factorials[num_features]
        )
        shap_values[player_index] = np.sum(
            weights * (included_outputs - excluded_outputs)
        )
    return shap_values


def _all_masks(num_players: int) -> tuple[tuple[bool, ...], ...]:
    return tuple(tuple(bits) for bits in product([False, True], repeat=num_players))


def _additive_payoffs(
    masks: tuple[tuple[bool, ...], ...],
    coefficients: tuple[float, ...],
) -> list[float]:
    return [
        sum(coef for coef, present in zip(coefficients, mask, strict=True) if present)
        for mask in masks
    ]


def test_iter_mask_ints_covers_full_table() -> None:
    """Bitmask enumeration yields every coalition from empty to grand."""
    assert set(ExactEstimator.iter_mask_ints(3)) == set(range(8))


def test_num_coalitions_is_full_table() -> None:
    """Exact enumeration evaluates every coalition including empty and grand."""
    assert ExactEstimator.num_coalitions(1) == 2
    assert ExactEstimator.num_coalitions(3) == 8


def test_sample_masks_includes_grand_coalition() -> None:
    """sample_masks yields the all-present coalition for backend evaluation."""
    player_set = PlayerSet(player_ids=("p0", "p1", "p2"))
    masks = list(
        ExactEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=False,
            seed=0,
        )
    )
    assert len(masks) == 8
    assert any(all(mask.present) for mask in masks)


def test_iter_masks_is_lazy() -> None:
    """iter_masks yields masks without materializing the full coalition set."""
    player_set = PlayerSet(player_ids=tuple(f"p{index}" for index in range(16)))
    masks = ExactEstimator.iter_masks(player_set)
    first = next(masks)
    assert isinstance(first, CoalitionMask)
    assert first.present == (False,) * player_set.num_players


def test_sample_masks_covers_all_coalitions() -> None:
    """Enumerated masks match every coalition including the grand coalition."""
    player_set = PlayerSet(player_ids=("a", "b"))
    masks = list(
        ExactEstimator().sample_masks(
            player_set,
            budget_fraction=1.0,
            include_minimal_masks=True,
            seed=99,
        )
    )
    expected = set(product([False, True], repeat=2))
    assert {mask.present for mask in masks} == expected


def test_sample_masks_requires_full_budget() -> None:
    """Partial budgets are incompatible with exact enumeration."""
    player_set = PlayerSet(player_ids=("p0",))
    with pytest.raises(ValueError, match="budget_fraction"):
        ExactEstimator().sample_masks(
            player_set,
            budget_fraction=0.5,
            include_minimal_masks=False,
            seed=0,
        )


def test_shapley_matches_legacy_additive_games() -> None:
    """Shapley attributions match legacy precise values on linear games."""
    for num_players in (2, 3):
        coefficients = tuple(float(index + 1) for index in range(num_players))
        masks = _all_masks(num_players)
        payoffs = _additive_payoffs(masks, coefficients)
        coalition_masks = tuple(CoalitionMask.from_sequence(mask) for mask in masks)

        estimated = ExactEstimator().estimate_attributions(
            coalition_masks,
            payoffs,
            ShapleyAggregator(),
        )
        reference = _reference_shapley_values(
            np.asarray(masks, dtype=bool),
            np.asarray(payoffs, dtype=np.float64),
        )
        assert estimated == pytest.approx(reference.tolist(), rel=1e-5, abs=1e-5)


def test_shapley_efficiency_property() -> None:
    """Shapley values sum to the total characteristic-function change."""
    masks = _all_masks(2)
    payoffs = [0.0, 1.0, 2.0, 3.5]
    coalition_masks = tuple(CoalitionMask.from_sequence(mask) for mask in masks)
    values = ExactEstimator().estimate_attributions(
        coalition_masks,
        payoffs,
        ShapleyAggregator(),
    )
    assert sum(values) == pytest.approx(payoffs[-1] - payoffs[0], abs=1e-6)


def test_estimate_attributions_delegates_to_estimand() -> None:
    """Majority-game attributions diverge across Shapley and Banzhaf estimands."""
    masks = _all_masks(3)

    def majority_payoff(mask: tuple[bool, ...]) -> float:
        return 1.0 if sum(mask) >= 2 else 0.0

    payoffs = [majority_payoff(mask) for mask in masks]
    coalition_masks = tuple(CoalitionMask.from_sequence(mask) for mask in masks)
    estimator = ExactEstimator()

    shapley = estimator.estimate_attributions(
        coalition_masks,
        payoffs,
        ShapleyAggregator(),
    )
    banzhaf = estimator.estimate_attributions(
        coalition_masks,
        payoffs,
        BanzhafAggregator(),
    )

    assert shapley != banzhaf
    assert shapley == pytest.approx([2 / 6, 2 / 6, 2 / 6])
    assert banzhaf == pytest.approx([0.5, 0.5, 0.5])


def test_exact_estimator_exposes_name() -> None:
    """Estimator reports the registered plugin identifier."""
    assert ExactEstimator().name == "exact"


def test_aggregators_used_by_exact_report_estimand_labels() -> None:
    """Wired estimands keep their public labels."""
    assert ShapleyAggregator().estimand is Estimand.SHAPLEY
    assert BanzhafAggregator().estimand is Estimand.BANZHAF
