"""Absence policies and partitioners on a real model.

@see [[src/nlp_shap/masking/policies.py]]
"""

from typing import Any, cast

import pytest

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.plugins import PluginContext, PluginGroup, default_registry
from nlp_shap.protocols.masking import AbsencePolicy
from nlp_shap.protocols.partition import PlayerPartitioner
from tests.practical._harness import (
    SCENARIOS,
    PracticalModel,
    Scenario,
    labels,
    scenario_cases,
    summary,
)
from tests.practical._judge import top_players


def _empty_coalition_text(scenario: Scenario, model: PracticalModel) -> str:
    """Render the empty coalition with the scenario's partition and policy."""
    config = scenario.config(model.spec)
    registry = default_registry()
    context = PluginContext(config=config, registry=registry)
    explanation = config.explanation
    players = cast(
        PlayerPartitioner,
        registry.resolve(PluginGroup.PARTITIONS, explanation.players, context),
    )
    policy = cast(
        AbsencePolicy,
        registry.resolve(
            PluginGroup.ABSENCE_POLICIES, explanation.absence_policy, context
        ),
    )
    player_set = players.partition(scenario.snapshot)
    empty = CoalitionMask.from_sequence((False,) * player_set.num_players)
    return policy.apply(scenario.snapshot, player_set, empty).messages()[0].text


@pytest.mark.parametrize(
    "scenario", scenario_cases("masking.delete", "masking.pad", "masking.neutral")
)
def test_policies_keep_grand_value_and_top_player(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """The empty coalition renders as specified; v(N) and the top player hold."""
    output = default_model.explain(scenario)
    reference = default_model.explain(SCENARIOS["masking.delete"])
    practical_record["numbers"] = summary(output)

    assert (
        _empty_coalition_text(scenario, default_model) == scenario.expect["empty_text"]
    )
    assert output.result.grand_value == pytest.approx(reference.result.grand_value)
    top = top_players(labels(output), output.result.values)
    assert scenario.expect["top_player"] in top


@pytest.mark.parametrize("scenario", scenario_cases("partition.tokens"))
def test_token_players_are_whitespace_tokens(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """One player per whitespace token, labelled with the token."""
    output = default_model.explain(scenario)
    practical_record["numbers"] = summary(output)

    assert labels(output) == scenario.expect["labels"]


@pytest.mark.parametrize("scenario", scenario_cases("partition.messages"))
def test_message_players_follow_the_roles_option(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Only user messages are players; both drive the answer, the question most."""
    output = default_model.explain(scenario)
    names = labels(output)
    practical_record["numbers"] = summary(output)

    assert len(names) == scenario.expect["players"]
    assert all(name.startswith("user ") for name in names)
    assert all(output.result.values[i] > 0 for i in scenario.expect["positive"])
    expected = names[scenario.expect["top_player"]]
    assert top_players(names, output.result.values) == [expected]
