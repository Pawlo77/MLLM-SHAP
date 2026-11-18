"""Removal metrics: brute force on the reference games, win rates, flips, circularity.

@see [[src/mllm_shap/pipeline/faithfulness.py]]
"""

import asyncio
import logging
from pathlib import Path

import numpy as np
import pytest

from mllm_shap import ExplainConfig
from mllm_shap.pipeline.faithfulness import (
    compute_removal_metrics,
    removal_metrics,
    removal_plan,
)
from tests.backends.test_replay import _live as replay_live
from tests.backends.test_replay import _replay as replay_config
from tests.backends.test_replay import _snapshot as replay_snapshot
from tests.estimation._games import Game, additive_game, reference_games
from tests.pipeline.test_pipeline_axioms import _registry, _snapshot


@pytest.mark.parametrize("game", reference_games(8), ids=lambda game: game.name)
def test_values_equal_brute_force_on_the_reference_games(game: Game) -> None:
    ranking = game.shapley
    metrics = compute_removal_metrics(ranking, game.payoff)
    order = np.argsort(-ranking, kind="stable")
    n = game.num_players
    for k, deleted, inserted in zip(
        metrics.ks, metrics.deletion_values, metrics.insertion_values, strict=True
    ):
        top = set(order[:k].tolist())
        assert deleted == pytest.approx(
            game.payoff(tuple(i not in top for i in range(n)))
        )
        assert inserted == pytest.approx(game.payoff(tuple(i in top for i in range(n))))
    assert metrics.comprehensiveness(n) == pytest.approx(
        metrics.grand_value - metrics.empty_value
    )
    assert metrics.sufficiency(n) == pytest.approx(0.0)


def test_aopc_is_one_when_the_first_removal_reaches_the_empty_value() -> None:
    metrics = compute_removal_metrics(np.ones(6), lambda mask: float(all(mask)))
    assert metrics.aopc == pytest.approx(1.0)
    flat = compute_removal_metrics(np.ones(6), lambda mask: 1.0)
    assert flat.aopc == 0.0


def test_exact_values_win_every_random_comparison_on_an_additive_game() -> None:
    game = additive_game(10)
    metrics = compute_removal_metrics(game.shapley, game.payoff, random_sets=30, seed=1)
    assert metrics.deletion_win_rate == pytest.approx(1.0, abs=0.02)
    assert metrics.insertion_win_rate == pytest.approx(1.0, abs=0.02)


def test_shuffled_values_win_about_half() -> None:
    game = additive_game(10)
    rng = np.random.default_rng(0)
    rates = []
    for seed in range(40):
        shuffled = rng.permutation(game.shapley)
        metrics = compute_removal_metrics(
            shuffled, game.payoff, random_sets=20, seed=seed
        )
        rates.append(metrics.deletion_win_rate)
    assert float(np.mean(rates)) == pytest.approx(0.5, abs=0.07)


def test_without_random_sets_there_is_no_win_rate() -> None:
    metrics = compute_removal_metrics([1.0, 0.0, 0.5], lambda mask: float(sum(mask)))
    assert (metrics.deletion_win_rate, metrics.insertion_win_rate) == (None, None)


def test_flip_and_restore_sizes_follow_the_ranking() -> None:
    # The decision holds while player 0, or both of players 1 and 2, are present.
    def payoff(mask: tuple[bool, ...]) -> float:
        return 1.0 if mask[0] or (mask[1] and mask[2]) else -1.0

    good = compute_removal_metrics([3.0, 2.0, 1.0], payoff, threshold=0.0)
    assert (good.flip_size, good.restore_size) == (2, 1)
    bad = compute_removal_metrics([1.0, 2.0, 3.0], payoff, threshold=0.0)
    assert (bad.flip_size, bad.restore_size) == (3, 2)
    flipped_already = compute_removal_metrics(
        [1.0, 2.0], lambda mask: -1.0, threshold=0.0
    )
    assert flipped_already.flip_size is None


def test_plan_is_unique_seeded_and_includes_the_boundaries() -> None:
    plan = removal_plan([0.3, 0.2, 0.1, 0.0], random_sets=5, seed=2)
    assert len(plan) == len(set(plan))
    assert plan == removal_plan([0.3, 0.2, 0.1, 0.0], random_sets=5, seed=2)
    assert (True,) * 4 in plan
    assert (False,) * 4 in plan
    with pytest.raises(ValueError, match="0 to 4"):
        removal_plan([0.0] * 4, ks=(5,))


def test_matching_value_function_and_filler_is_flagged(caplog) -> None:
    same = ("tfidf_cosine", "delete")
    with caplog.at_level(logging.WARNING):
        flagged = compute_removal_metrics(
            [1.0, 0.0],
            lambda mask: float(sum(mask)),
            explained_with=same,
            evaluated_with=same,
        )
    assert flagged.circular
    assert "favour the explainer" in caplog.text
    other = compute_removal_metrics(
        [1.0, 0.0],
        lambda mask: float(sum(mask)),
        explained_with=same,
        evaluated_with=("logprob", "delete"),
    )
    assert not other.circular


def test_pipeline_run_scores_the_plan_once_and_flags_circularity() -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "echo", "model_id": "echo"},
        "explanation": {"value_fn": "count_test", "archive": {"path": ""}},
    })
    snapshot = _snapshot("alpha beta gamma delta")
    metrics = asyncio.run(
        removal_metrics(
            config,
            snapshot,
            [0.9, 0.1, 0.5, -0.2],
            random_sets=4,
            explained_with=config,
            registry=_registry(),
        )
    )
    assert metrics.deletion_values == pytest.approx((2.0, 1.0, 0.0, 0.0, 0.0))
    assert metrics.insertion_values == pytest.approx((0.0, 1.0, 2.0, 2.0, 2.0))
    assert metrics.circular
    assert metrics.deletion_win_rate is not None


def test_the_curve_areas_are_trapezoids_over_the_removed_fraction() -> None:
    metrics = compute_removal_metrics([2.0, 1.0], lambda mask: float(sum(mask)))
    assert metrics.deletion_values == (2.0, 1.0, 0.0)
    assert metrics.deletion_area == pytest.approx(1.0)
    assert metrics.insertion_area == pytest.approx(1.0)


def test_replay_answers_the_removal_run_from_an_archive(tmp_path: Path) -> None:
    attributions = [0.5, 0.2, 0.1, 0.0, -0.1, 0.3]
    snapshot = replay_snapshot(len(attributions))
    live = asyncio.run(
        removal_metrics(
            replay_live(tmp_path / "live"), snapshot, attributions, random_sets=5
        )
    )
    # Without a fallback backend, one unarchived coalition would raise ReplayMissError.
    replayed = asyncio.run(
        removal_metrics(
            replay_config(tmp_path / "again", tmp_path / "live"),
            snapshot,
            attributions,
            random_sets=5,
        )
    )
    assert replayed == live
