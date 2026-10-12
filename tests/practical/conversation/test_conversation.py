"""Multi-turn, multi-user, system-prompt and audio conversations.

@see [[src/mllm_shap/masking/partitions.py]]
@see [[src/mllm_shap/backends/transformers/audio.py]]
"""

import asyncio
from pathlib import Path
from typing import Any

import pytest

from mllm_shap import ExplainRunner
from mllm_shap.backends.lmstudio import LmStudioBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.pipeline.config import BackendConfig, ExplainConfig
from mllm_shap.plugins import PluginContext
from tests.practical._harness import (
    MODELS,
    PracticalModel,
    Scenario,
    labels,
    require_weights,
    scenario_cases,
    summary,
)
from tests.practical._judge import driver_prompts, top_players

ROOT = Path(__file__).parents[3]


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
def test_lmstudio_refuses_audio_by_default(scenario: Scenario) -> None:
    """The LM Studio SDK takes no audio: the backend raises before any call."""
    config = BackendConfig(kind="lmstudio", model_id="unused")
    backend = LmStudioBackend(PluginContext(config=ExplainConfig(backend=config)))
    with pytest.raises(ValueError, match="audio: transcript"):
        asyncio.run(backend.generate(_audio_snapshot(scenario), GenerationConfig()))


@pytest.mark.parametrize("scenario", scenario_cases("conversation.audio"))
def test_audio_model_hears_the_recording_and_explains_its_words(
    scenario: Scenario,
    audio_model: PracticalModel,
    practical_record: dict[str, Any],
) -> None:
    """Granite transcribes the clip from audio alone; spoken words get values."""
    pytest.importorskip("librosa")
    pytest.importorskip("torchaudio")
    require_weights(MODELS["alignment"])
    config = scenario.config(MODELS[scenario.tier])
    audio = AudioPart(ROOT / scenario.expect["audio"], sample_rate=16_000)
    ask = Message(role=Role.USER, content=(audio, TextPart(scenario.expect["ask"])))
    backend = audio_model.backend
    heard = asyncio.run(
        backend.generate(
            ConversationSnapshot.from_turns((Turn(messages=(ask,)),)),
            config.generation.model_copy(
                update={"max_new_tokens": scenario.expect["heard_tokens"]}
            ),
        )
    )
    transcript = scenario.snapshot.messages()[0].text
    spoken = Message(role=Role.USER, content=(TextPart(transcript), audio))
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(spoken,)),))
    practical_record["properties"] = {"heard": heard.text}
    practical_record["numbers"] = {}
    assert scenario.expect["heard"] in heard.text.lower()
    # The base config sends the transcript too; the variant (align_only) sends audio.
    for variant in [{}, *scenario.variants]:
        run_config = scenario.config(MODELS[scenario.tier], variant)
        output = ExplainRunner(run_config, backend=backend).explain_sync(snapshot)
        result = output.result
        assert output.player_set is not None
        spans = output.player_set.spans
        mode = "align_only" if variant else "send"
        practical_record["numbers"][mode] = {
            span.text: value for span, value in zip(spans, result.values, strict=True)
        }
        practical_record["properties"][f"{mode}_efficiency_gap"] = sum(
            result.values
        ) - (result.grand_value - result.base_value)

        assert len(spans) == len(transcript.split())
        assert all(span.samples is not None for span in spans)
        assert max(abs(value) for value in result.values) > 0.0
