"""Leverage SHAP: leverage scores, allocation, exactness, efficiency and accuracy.

@see [[src/mllm_shap/estimation/leverage.py]]
"""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation._shared import budget_fraction
from mllm_shap.estimation.complementary import ComplementaryEstimator
from mllm_shap.estimation.estimands import BanzhafAggregator, ShapleyAggregator
from mllm_shap.estimation.leverage import (
    LeverageEstimator,
    allocate_pairs,
    distinct_subsets,
)
from tests.estimation._games import Game, reference_games


def _run(
    game: Game, estimator_type: type, fraction: float, seed: int
) -> tuple[list[float], list[tuple[bool, ...]]]:
    players = PlayerSet(player_ids=tuple(f"p{i}" for i in range(game.num_players)))
    estimator = estimator_type()
    masks = list(estimator.sample_masks(players, fraction, False, seed))
    values = estimator.estimate_attributions(
        masks, [game.payoff(m.present) for m in masks], ShapleyAggregator()
    )
    return values, [m.present for m in masks]


def test_every_size_has_the_same_total_leverage() -> None:
    """SVD leverage of the Kernel SHAP system: score times stratum size is 1."""
    num_players = 6
    rows = []
    sizes = []
    for mask in range(1, 2**num_players - 1):
        z = np.array([(mask >> i) & 1 for i in range(num_players)], dtype=float)
        size = int(z.sum())
        kernel = (num_players - 1) / (
            math.comb(num_players, size) * size * (num_players - size)
        )
        rows.append((z[:-1] - z[-1]) * np.sqrt(kernel))
        sizes.append(size)
    q, _ = np.linalg.qr(np.array(rows))
    leverage = (q**2).sum(axis=1)
    for size in range(1, num_players):
        index = sizes.index(size)
        assert leverage[index] * math.comb(num_players, size) == pytest.approx(1.0)


def test_allocation_enumerates_covered_classes_and_splits_the_rest() -> None:
    # n = 6: pair classes of sizes 1, 2, 3 hold 6, 15 and 10 pairs.
    assert allocate_pairs(6, 31) == {1: 6, 2: 15, 3: 10}
    assert allocate_pairs(6, 100) == {1: 6, 2: 15, 3: 10}
    # Units 2, 2, 1: 10 pairs split 4, 4, 2; size 1 (6 pairs) is not covered.
    assert allocate_pairs(6, 10) == {1: 4, 2: 4, 3: 2}
    # Size 1 is covered (share 12 * 2/5 = 4.8 < 6 is not; 15 pairs: 6 <= 6).
    allocation = allocate_pairs(6, 20)
    assert allocation[1] == 6
    assert sum(allocation.values()) == 20


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_exact_at_full_budget(game: Game) -> None:
    values, _ = _run(game, LeverageEstimator, 1.0, 0)
    assert values == pytest.approx(game.shapley, abs=1e-9)


@pytest.mark.parametrize("multiple", [2, 4, 8, 32])
def test_efficient_at_every_budget(multiple: int) -> None:
    for game in reference_games(10):
        values, masks = _run(
            game, LeverageEstimator, budget_fraction(10, None, multiple * 10), 1
        )
        assert sum(values) == pytest.approx(
            game.payoff((True,) * 10) - game.payoff((False,) * 10)
        )
        assert len(masks) == len(set(masks))


def test_sampler_emits_boundary_then_complementary_pairs() -> None:
    game = reference_games(10)[0]
    _, masks = _run(game, LeverageEstimator, budget_fraction(10, None, 60), 3)
    assert masks[:2] == [(True,) * 10, (False,) * 10]
    assert len(masks) == 60
    for left, right in zip(masks[2::2], masks[3::2], strict=True):
        assert right == tuple(not value for value in left)
    assert _run(game, LeverageEstimator, budget_fraction(10, None, 60), 3)[1] == masks


def test_rejects_the_banzhaf_estimand() -> None:
    players = PlayerSet(player_ids=("a", "b", "c"))
    estimator = LeverageEstimator()
    masks = list(estimator.sample_masks(players, 1.0, False, 0))
    with pytest.raises(ValueError, match="shapley estimand only"):
        estimator.estimate_attributions(masks, [0.0] * len(masks), BanzhafAggregator())
    with pytest.raises(ValidationError, match="leverage"):
        ExplainConfig.model_validate({
            "backend": {"kind": "mock", "model_id": "mock"},
            "explanation": {"estimator": "leverage", "estimand": "banzhaf"},
        })


@pytest.mark.parametrize("name", ["additive", "pairwise", "unanimity"])
@pytest.mark.parametrize("multiple", [16, 32])
def test_beats_complementary_sampling(name: str, multiple: int) -> None:
    game = next(g for g in reference_games(10) if g.name == name)
    fraction = budget_fraction(10, None, multiple * 10)
    errors = {}
    for estimator_type in (LeverageEstimator, ComplementaryEstimator):
        errors[estimator_type] = np.mean([
            np.linalg.norm(
                np.array(_run(game, estimator_type, fraction, seed)[0]) - game.shapley
            )
            for seed in range(20)
        ])
    assert errors[LeverageEstimator] < errors[ComplementaryEstimator]


def test_reanalyze_reproduces_the_run(tmp_path: Path) -> None:
    text = " ".join(f"w{i}" for i in range(8))
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "estimator": "leverage",
            "budget": {"per_player": 4},
            "absence_policy": "pad",
            "archive": {"path": str(tmp_path / "{run_id}")},
        },
    })
    runner = ExplainRunner(config)
    first = runner.explain_sync(snapshot)
    again = runner.reanalyze(tmp_path / first.run_id)
    assert again.result.values == pytest.approx(first.result.values, abs=1e-12)
    assert sum(first.result.values) == pytest.approx(
        first.result.grand_value - first.result.base_value
    )


def test_single_player_gets_the_whole_gap() -> None:
    estimator = LeverageEstimator()
    players = PlayerSet(player_ids=("a",))
    masks = list(estimator.sample_masks(players, 1.0, False, 0))
    values = estimator.estimate_attributions(masks, [3.0, 1.0], ShapleyAggregator())
    assert values == [2.0]


def test_boundary_coalitions_are_required() -> None:
    players = PlayerSet(player_ids=("a", "b", "c"))
    estimator = LeverageEstimator()
    masks = list(estimator.sample_masks(players, 1.0, False, 0))[2:]
    with pytest.raises(ValueError, match="grand and the empty"):
        estimator.estimate_attributions(masks, [0.0] * len(masks), ShapleyAggregator())


def test_distinct_subsets_redraws_repeats() -> None:
    """Rejection sampling (large populations) skips a subset it already yielded."""

    class Repeating:
        def __init__(self) -> None:
            self.calls = 0

        def choice(self, population: int, size: int, replace: bool) -> list[int]:
            self.calls += 1
            return list(range(size)) if self.calls < 3 else list(range(1, size + 1))

    rng = Repeating()
    drawn = list(distinct_subsets(rng, 16, 8, 2, contains_first=False))  # type: ignore[arg-type]
    assert len(drawn) == 2
    assert drawn[0] != drawn[1]
    assert rng.calls == 3
