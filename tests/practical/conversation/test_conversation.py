"""Multi-turn, multi-user, system-prompt and audio conversations.

@see [[src/nlp_shap/masking/partitions.py]]
"""

import asyncio
from typing import Any

import pytest

from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from nlp_shap.domain.enums import Role
from nlp_shap.pipeline.config import BackendConfig, ExplainConfig
from nlp_shap.plugins import PluginContext, PluginGroup
from nlp_shap.plugins.builtin import BUILTIN_PLUGINS
from tests.practical._harness import (
    PracticalModel,
    Scenario,
    labels,
    scenario_cases,
    summary,
)
from tests.practical._judge import driver_prompts, top_players


def _conversation(scenario: Scenario) -> str:
    return "\n".join(f"{m.role.value}: {m.text}" for m in scenario.snapshot.messages())


@pytest.mark.parametrize("scenario", scenario_cases("conversation.multi_turn"))
def test_name_message_outranks_the_distractor(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Across three turns the message carrying the name drives the answer."""
    output = default_model.explain(scenario)
    values, names = output.result.values, labels(output)
    answer = default_model.answer(scenario)
    top = top_players(names, values)[0]
    verdict = default_model.judge.ask(
        driver_prompts(_conversation(scenario), answer, top)
    )
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {"answer": answer}
    practical_record["judge"]["top_span_plausible"] = vars(verdict)

    assert scenario.expect["answer"] in answer
    assert values[scenario.expect["carrier"]] > values[scenario.expect["distractor"]]
    assert verdict.outcome != "fail", verdict


@pytest.mark.parametrize("scenario", scenario_cases("conversation.multi_user"))
def test_speaker_message_outranks_the_other_speaker(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Bob's order outranks Alice's; labels name the speakers."""
    output = default_model.explain(scenario)
    values, names = output.result.values, labels(output)
    answer = default_model.answer(scenario)
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {"answer": answer}

    assert scenario.expect["answer"] in answer.lower()
    assert values[scenario.expect["bob"]] > values[scenario.expect["alice"]]
    assert names[0].startswith("user/Alice 1:")
    assert names[1].startswith("user/Bob 2:")


@pytest.mark.parametrize("scenario", scenario_cases("conversation.system_prompt"))
def test_system_prompt_drives_the_language(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """The system message gets a positive value and the answer is French."""
    output = default_model.explain(scenario)
    answer = default_model.answer(scenario)
    verdict = default_model.judge.ask([
        f'Is the following text written in French?\n"{answer}"',
        f'Text: "{answer}"\nIs this text in the French language?',
        f'Would a reader say "{answer}" is French rather than English?',
    ])
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {"answer": answer}
    practical_record["judge"]["answer_is_french"] = vars(verdict)

    assert output.result.values[scenario.expect["system"]] > 0.0
    assert verdict.outcome != "fail", verdict


def _audio_snapshot(scenario: Scenario) -> ConversationSnapshot:
    text = scenario.snapshot.messages()[0].text
    audio = AudioPart(source=b"RIFF0000WAVE", sample_rate=16_000)
    message = Message(role=Role.USER, content=(TextPart(text), audio))
    return ConversationSnapshot.from_turns((Turn(messages=(message,)),))


@pytest.mark.parametrize("scenario", scenario_cases("conversation.audio"))
def test_audio_is_refused_by_every_real_backend(scenario: Scenario) -> None:
    """Text backends raise ValueError on an audio part before any model call."""
    backends = BUILTIN_PLUGINS[PluginGroup.BACKENDS]
    for kind, factory in backends.items():
        if kind == "mock":
            continue
        config = BackendConfig(
            kind=kind, model_id="unused", api_host="http://127.0.0.1:9/v1"
        )
        backend = factory(PluginContext(config=ExplainConfig(backend=config)))
        with pytest.raises(ValueError, match="audio"):
            asyncio.run(backend.generate(_audio_snapshot(scenario), GenerationConfig()))


@pytest.mark.parametrize("scenario", scenario_cases("conversation.audio"))
def test_audio_explanation(scenario: Scenario) -> None:
    """No backend generates from audio yet."""
    pytest.skip("no audio backend yet")
