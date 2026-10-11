"""Deep tier: every value function follows a swapped fact across a prompt pair.

@see [[src/mllm_shap/value/logprob.py]]
"""

import dataclasses
import math
from typing import Any

import pytest

from tests.practical._harness import PracticalModel, Scenario, scenario_cases
from tests.practical._judge import top_by_magnitude


def _swapped(scenario: Scenario, old: str, new: str) -> Scenario:
    turns = [
        [{**message, "text": message["text"].replace(old, new)} for message in turn]
        for turn in scenario.turns
    ]
    return dataclasses.replace(scenario, turns=turns)


def _day_prompts(answer: str, day: str) -> list[str]:
    return [
        f'Reply: "{answer}"\nDoes the reply say the meeting is on {day}?',
        f'Someone answered "{answer}". Does that answer name {day} as the day?',
        f'Is {day} the meeting day according to this text: "{answer}"?',
    ]


@pytest.mark.parametrize("scenario", scenario_cases("deep.value.counterfactual"))
def test_swapped_fact_keeps_its_role(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """The swapped token's sign and rank-1 status agree across the pair."""
    days, token = scenario.expect["swap"], scenario.expect["token"]
    pair = (scenario, _swapped(scenario, *days))
    rows: dict[str, dict[str, list[Any]]] = {}
    for variant in [{}, *scenario.variants]:
        explanation = variant.get("explanation", {})
        runs = [deep_model.explain(s, variant).result.values for s in pair]
        rows[explanation.get("value_fn", "tfidf_cosine")] = {
            "values": [[round(v, 4) for v in run] for run in runs],
            "sign": [math.copysign(1.0, run[token]) for run in runs],
            "rank1": [top_by_magnitude(run, 1)[0] == token for run in runs],
        }
    answers = [deep_model.answer(s) for s in pair]
    verdicts = [
        deep_model.judge.ask(_day_prompts(answer, day))
        for answer, day in zip(answers, days, strict=True)
    ]
    practical_record["tables"] = {"counterfactual": rows}
    practical_record["properties"] = {"answers": answers}
    for day, verdict in zip(days, verdicts, strict=True):
        practical_record["judge"][f"answer_says_{day}"] = vars(verdict)

    for name, row in rows.items():
        assert row["sign"][0] == row["sign"][1], (name, row)
        assert row["rank1"][0] == row["rank1"][1], (name, row)
    assert all(verdict.outcome != "fail" for verdict in verdicts), verdicts
