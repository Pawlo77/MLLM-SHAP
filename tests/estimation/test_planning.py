"""Budget planning from a target error.

@see [[src/mllm_shap/estimation/planning.py]]
"""

import math

import numpy as np
import pytest

from mllm_shap.domain.players import PlayerSet
from mllm_shap.estimation.estimands import ShapleyAggregator
from mllm_shap.estimation.permutation import PermutationEstimator
from mllm_shap.estimation.planning import plan_budget
from tests.estimation._games import reference_games


def test_orderings_follow_the_hoeffding_union_bound() -> None:
    plan = plan_budget(20, value_range=2.0, target_error=0.1, confidence=0.95)
    expected = math.ceil(4.0 * math.log(2.0 * 20 / 0.05) / (2.0 * 0.01))
    assert plan.orderings == expected
    assert plan.evaluations == min(expected * 19 + 2, 1 << 20)


def test_halving_the_error_quadruples_the_orderings() -> None:
    loose = plan_budget(12, 1.0, 0.2)
    tight = plan_budget(12, 1.0, 0.1)
    assert tight.orderings == pytest.approx(4 * loose.orderings, rel=0.01)


def test_a_loose_target_on_few_players_is_cheaper_than_the_full_table() -> None:
    plan = plan_budget(6, value_range=1.0, target_error=0.5)
    assert plan.evaluations <= 1 << 6
    tight = plan_budget(6, value_range=1.0, target_error=0.01)
    assert tight.evaluations == 1 << 6
    assert tight.exact_is_cheaper


@pytest.mark.parametrize(
    ("players", "value_range", "target", "confidence"),
    [(0, 1.0, 0.1, 0.95), (4, 0.0, 0.1, 0.95), (4, 1.0, 0.0, 0.95), (4, 1.0, 0.1, 1.0)],
)
def test_bad_inputs_are_refused(
    players: int, value_range: float, target: float, confidence: float
) -> None:
    with pytest.raises(ValueError, match="must"):
        plan_budget(players, value_range, target, confidence)


def test_the_planned_budget_reaches_the_target_error_on_a_reference_game() -> None:
    n = 12
    game = reference_games(n)[1]
    masks, payoffs = game.table()
    table = dict(zip(masks, payoffs, strict=True))
    marginals = [
        table[(*mask[:i], True, *mask[i + 1 :])] - table[mask]
        for mask in masks
        for i in range(n)
        if not mask[i]
    ]
    marginal_range = float(max(marginals) - min(marginals))
    target = 0.2 * marginal_range
    plan = plan_budget(n, value_range=marginal_range, target_error=target)
    assert not plan.exact_is_cheaper
    players = PlayerSet(player_ids=tuple(map(str, range(n))))
    worst = 0.0
    for seed in range(10):
        drawn = list(
            PermutationEstimator().sample_masks(
                players, plan.evaluations / (1 << n), False, seed
            )
        )
        values = ShapleyAggregator().aggregate(
            [m.present for m in drawn], [game.payoff(m.present) for m in drawn]
        )
        worst = max(worst, float(np.max(np.abs(np.array(values) - game.shapley))))
    assert worst <= target


def test_the_cli_prints_the_plan(capsys: pytest.CaptureFixture[str]) -> None:
    from mllm_shap.cli import main

    code = main(["plan", "--players", "20", "--range", "2", "--target-error", "0.1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "orderings:" in out
    assert "evaluations:" in out
    assert (
        main(["plan", "--players", "4", "--range", "-1", "--target-error", "0.1"]) != 0
    )


def test_the_cli_hints_at_exact_when_sampling_is_not_cheaper(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mllm_shap.cli import main

    assert (
        main(["plan", "--players", "6", "--range", "1", "--target-error", "0.01"]) == 0
    )
    assert "use estimator exact" in capsys.readouterr().out
