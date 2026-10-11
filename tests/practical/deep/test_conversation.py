"""Deep tier: a 10-turn, three-speaker thread and a per-speaker fact question.

@see [[src/mllm_shap/masking/partitions.py]]
"""

from collections.abc import Sequence
from typing import Any

import pytest

from tests.practical._deep import REGISTRY
from tests.practical._harness import (
    PracticalModel,
    Scenario,
    labels,
    scenario_cases,
    summary,
)


def _ranked(values: Sequence[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: -values[i])[:k]


def _rule_prompts(rule: str, answer: str) -> list[str]:
    # The 3B judge answers NO to "does the reply follow the rule?" for a compliant
    # reply (margins -2.4, -17.7, -3.2); these three separate by more than 8 nats.
    return [
        f'Does the text "{answer}" end with the word DONE?',
        f'Is the reply "{answer}" compliant with the rule "{rule}"?',
        (
            f'Someone was told: "{rule}" They wrote: "{answer}" '
            "Did they end with the word DONE?"
        ),
    ]


@pytest.mark.parametrize("scenario", scenario_cases("deep.conversation.long"))
def test_decisive_message_and_fact_token_lead(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """The fact message is in the top 2 messages, its fact token in the top 3."""
    expect = scenario.expect
    messages = deep_model.explain(scenario)
    tokens = deep_model.explain(scenario, scenario.variants[0], registry=REGISTRY)
    answer = deep_model.answer(scenario)
    rule = "Always end every reply with the single word DONE."
    verdict = deep_model.judge.ask(_rule_prompts(rule, answer))
    top_tokens = [  # labels read "<speaker>: <token>"
        labels(tokens)[i].split()[-1]
        for i in _ranked(tokens.result.values, expect["top_tokens"])
    ]
    practical_record["numbers"] = {
        "messages": summary(messages),
        "tokens": summary(tokens),
    }
    practical_record["properties"] = {
        "answer": answer,
        "ends_with_rule_word": answer.rstrip(" .!").endswith(expect["rule"]),
        "top_tokens": top_tokens,
    }
    practical_record["judge"]["answer_follows_rule"] = vars(verdict)

    assert expect["decisive"] in _ranked(messages.result.values, expect["top_messages"])
    assert expect["fact"] in top_tokens
    assert verdict.outcome != "fail", verdict


@pytest.mark.parametrize("scenario", scenario_cases("deep.conversation.speakers"))
def test_asked_speaker_ranks_first(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """The message of the speaker the question names ranks first; labels name them."""
    expect = scenario.expect
    output = deep_model.explain(scenario)
    names = labels(output)
    answer = deep_model.answer(scenario)
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {"answer": answer}

    assert expect["answer"] in answer
    assert _ranked(output.result.values, 1) == [expect["top"]]
    for name, speaker in zip(names, expect["speakers"], strict=True):
        assert name.startswith(f"user/{speaker} "), name
