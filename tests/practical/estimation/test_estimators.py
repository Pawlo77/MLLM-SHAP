"""Estimators and estimands on a real model against the exact reference.

@see [[src/mllm_shap/estimation/exact.py]]
@see [[src/mllm_shap/estimation/loo.py]]
"""

from typing import Any

import pytest

from tests.practical._harness import (
    SCENARIOS,
    PracticalModel,
    Scenario,
    labels,
    scenario_cases,
    summary,
)
from tests.practical._judge import (
    answer_prompts,
    efficiency_gap,
    normalised_l1,
    top_by_magnitude,
    top_players,
)


@pytest.mark.parametrize("scenario", scenario_cases("estimator.exact"))
def test_exact_reference(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Exact Shapley ranks France first, is efficient, and answers correctly."""
    output = default_model.explain(scenario)
    result = output.result
    gap = efficiency_gap(result.values, result.base_value, result.grand_value)
    answer = default_model.answer(scenario)
    verdict = default_model.judge.ask(
        answer_prompts(scenario.snapshot.messages()[0].text, answer)
    )
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {"efficiency_gap": gap, "answer": answer}
    practical_record["judge"]["answer_correct"] = vars(verdict)

    assert scenario.expect["top_player"] in top_players(labels(output), result.values)
    assert gap <= scenario.expect["efficiency"]
    assert verdict.outcome != "fail", verdict


@pytest.mark.parametrize(
    "scenario",
    scenario_cases("estimator.mc", "estimator.complementary", "estimator.neyman_cc"),
)
def test_sampled_estimators_track_exact(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """On a partial budget: exact's top player, top-3 signs, bounded L1 distance."""
    exact = default_model.explain(SCENARIOS["estimator.exact"])
    output = default_model.explain(scenario)
    names = labels(exact)
    reference, estimate = exact.result.values, output.result.values
    top = top_by_magnitude(reference, scenario.expect["top_k_signs"])
    distance = normalised_l1(estimate, reference)
    practical_record["numbers"] = {"exact": summary(exact), "estimate": summary(output)}
    practical_record["properties"] = {"normalised_l1": distance}

    assert set(top_players(names, estimate)) <= set(top_players(names, reference))
    assert [estimate[i] > 0 for i in top] == [reference[i] > 0 for i in top]
    assert distance <= scenario.expect["max_l1"]


@pytest.mark.parametrize("scenario", scenario_cases("estimator.loo"))
def test_leave_one_out_needs_n_plus_two_coalitions(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Leave-one-out ranks France first from the grand, empty and n drops."""
    output = default_model.explain(scenario)
    names = labels(output)
    practical_record["numbers"] = summary(output)

    assert scenario.expect["top_player"] in top_players(names, output.result.values)
    assert output.metrics.requested == len(names) + 2


@pytest.mark.parametrize("scenario", scenario_cases("estimand.banzhaf"))
def test_banzhaf_exact_and_sampled(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Exact Banzhaf ranks France first; Monte Carlo keeps it near the top."""
    exact = default_model.explain(scenario)
    sampled = default_model.explain(scenario, scenario.variants[0])
    names, reference = labels(exact), exact.result.values
    estimate = sampled.result.values
    expect = scenario.expect
    distance = normalised_l1(estimate, reference)
    practical_record["numbers"] = {"exact": summary(exact), "mc": summary(sampled)}
    practical_record["properties"] = {"normalised_l1": distance}

    assert expect["top_player"] in top_players(names, reference)
    nearest = top_by_magnitude(estimate, expect["mc_top_k"])
    assert names.index(expect["top_player"]) in nearest
    assert distance <= expect["max_l1"]
