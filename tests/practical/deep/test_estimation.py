"""Deep tier: sampled estimators and Banzhaf against a 10-player exact reference.

The 1024-coalition reference is computed once (generation cache and
``results/reference-<hash>.json``); every sampled run reuses its generations.

@see [[src/mllm_shap/estimation/neyman.py]]
"""

import statistics
from collections.abc import Sequence
from typing import Any

import pytest

from tests.practical._deep import (
    error_table,
    non_increasing,
    reference,
    sweep,
    top1_agreement,
)
from tests.practical._harness import PracticalModel, Scenario, scenario_cases
from tests.practical._judge import top_by_magnitude

PAIRED = ("complementary", "neyman_cc")
"""Samplers that must beat plain Monte Carlo."""


def _runs(
    model: PracticalModel, scenario: Scenario, budgets: Sequence[int], seeds: int
) -> dict[str, dict[int, list[list[float]]]]:
    return {
        variant["explanation"]["estimator"]: sweep(
            model, scenario, budgets, range(seeds), variant
        )
        for variant in scenario.variants
    }


@pytest.mark.parametrize("scenario", scenario_cases("deep.estimator.reference"))
def test_error_shrinks_with_budget(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Median L1 error is non-increasing; paired samplers beat mc at the first."""
    expect = scenario.expect
    budgets = expect["budgets"]
    exact = reference(deep_model, scenario)
    table = error_table(
        _runs(deep_model, scenario, budgets, expect["seeds"]), exact["values"]
    )
    wins = {
        method: {
            budget: sum(
                e < m
                for e, m in zip(
                    table[method][budget]["per_seed"],
                    table["mc"][budget]["per_seed"],
                    strict=True,
                )
            )
            for budget in budgets
        }
        for method in PAIRED
    }
    top3 = [exact["labels"][i] for i in top_by_magnitude(exact["values"], 3)]
    practical_record["numbers"] = {"exact": exact}
    practical_record["tables"] = {"error_vs_budget": table, "wins_over_mc": wins}
    practical_record["properties"] = {"top3": top3}

    assert set(expect["facts"]) <= set(top3)
    for method, by_budget in table.items():
        medians = [by_budget[b]["median"] for b in budgets]
        assert non_increasing(medians), (method, medians)
    for method, by_budget in wins.items():
        assert by_budget[budgets[0]] >= expect["beats_mc"], (method, by_budget)


@pytest.mark.parametrize("scenario", scenario_cases("deep.estimator.stability"))
def test_paired_samplers_agree_on_the_top_player(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """At one budget the paired samplers find exact's top player on most seeds."""
    expect = scenario.expect
    exact = reference(deep_model, scenario)["values"]
    runs = _runs(deep_model, scenario, [expect["budget"]], expect["seeds"])
    top3 = top_by_magnitude(exact, 3)
    table = {
        method: {
            "top1_agreement": top1_agreement(by_budget[expect["budget"]], exact),
            "top3_std": [
                round(statistics.stdev(v[i] for v in by_budget[expect["budget"]]), 4)
                for i in top3
            ],
        }
        for method, by_budget in runs.items()
    }
    practical_record["tables"] = {"stability": table}

    for method in PAIRED:
        assert table[method]["top1_agreement"] >= expect["top1_agreement"], method


@pytest.mark.parametrize("scenario", scenario_cases("deep.estimand.banzhaf"))
def test_banzhaf_mc_converges_to_exact(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Monte Carlo Banzhaf error shrinks with the budget and finds exact's top 3."""
    expect = scenario.expect
    budgets = expect["budgets"]
    exact = reference(deep_model, scenario)
    runs = sweep(
        deep_model, scenario, budgets, range(expect["seeds"]), scenario.variants[0]
    )
    table = error_table({"mc": runs}, exact["values"])
    top3 = set(top_by_magnitude(exact["values"], 3))
    hits = {b: sum(top_by_magnitude(v, 1)[0] in top3 for v in runs[b]) for b in budgets}
    medians = [table["mc"][b]["median"] for b in budgets]
    practical_record["numbers"] = {"exact": exact}
    practical_record["tables"] = {"error_vs_budget": table}
    practical_record["properties"] = {"top1_in_exact_top3": hits}

    assert non_increasing(medians), medians
    assert medians[-1] <= expect["max_median_l1"]
    assert hits[budgets[-1]] >= expect["top3_hits"]
