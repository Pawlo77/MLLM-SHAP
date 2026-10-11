"""Spoken words of a real recording as players, through the whole pipeline.

@see [[src/mllm_shap/alignment/partitions.py]]
"""

from pathlib import Path
from typing import Any

import pytest

from mllm_shap import ExplainRunner
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from tests.practical._harness import MODELS, Scenario, require_weights, scenario_cases

ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize("scenario", scenario_cases("partition.sgpa_segments"))
def test_every_word_is_an_audio_player_and_efficiency_holds(
    scenario: Scenario, practical_record: dict[str, Any]
) -> None:
    """Nine words become nine players with sample ranges; values add up."""
    pytest.importorskip("librosa")
    pytest.importorskip("transformers")
    require_weights(MODELS["alignment"])
    message = scenario.snapshot.messages()[0]
    audio = AudioPart(ROOT / scenario.expect["audio"], sample_rate=16000)
    spoken = Message(role=message.role, content=(TextPart(message.text), audio))
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(spoken,)),))
    config = scenario.config(MODELS[scenario.tier])
    output = ExplainRunner(config).explain_sync(snapshot)
    assert output.player_set is not None
    spans = output.player_set.spans
    practical_record["numbers"] = {
        span.text: list(span.samples or ()) for span in spans
    }

    assert [span.text for span in spans] == message.text.split()
    assert all(span.samples is not None for span in spans)
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value
    )
