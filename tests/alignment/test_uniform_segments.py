"""Audio players from equal-duration segments: the matched-count control of SGPA.

@see [[src/mllm_shap/alignment/partitions.py]]
"""

from itertools import pairwise

import pytest

pytest.importorskip("soundfile")

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.alignment.partitions import UniformOptions, UniformSegmentPartitioner
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from tests.alignment._fakes import FRAME, FRAMES, RATE, speech

LENGTH = len(FRAMES) * FRAME
GRID = RATE // 50  # 20 ms


def _snapshot(transcript: str = "ab — c") -> ConversationSnapshot:
    spoken = Message(
        role=Role.USER, content=(TextPart(transcript), AudioPart(speech(), RATE))
    )
    return ConversationSnapshot.from_turns((Turn(messages=(spoken,)),))


def test_one_segment_per_transcript_word_covers_the_audio_on_a_frame_grid() -> None:
    players = UniformSegmentPartitioner().partition(_snapshot())
    assert [span.text for span in players.spans] == ["ab", "—", "c"]
    ranges = [span.samples for span in players.spans]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == LENGTH
    assert all(a[1] == b[0] for a, b in pairwise(ranges))
    assert all(start % GRID == 0 for start, _ in ranges)
    sizes = [end - start for start, end in ranges]
    assert max(sizes) - min(sizes) <= 2 * GRID


def test_the_transcript_is_not_sent() -> None:
    players = UniformSegmentPartitioner().partition(_snapshot())
    assert players.hidden_text == (0,)


def test_segments_option_overrides_the_word_count() -> None:
    options = UniformOptions(segments=2)
    from mllm_shap.plugins.context import PluginContext

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock_audio", "model_id": "stub"},
        "explanation": {"players": "uniform_segments"},
    })
    context = PluginContext(config=config, options=options.model_dump())
    players = UniformSegmentPartitioner(context).partition(_snapshot())
    assert len(players.player_ids) == 2


def test_more_segments_than_frames_is_refused() -> None:
    from mllm_shap.plugins.context import PluginContext

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock_audio", "model_id": "stub"},
        "explanation": {"players": "uniform_segments"},
    })
    context = PluginContext(config=config, options={"segments": 1000})
    with pytest.raises(ValueError, match="frames"):
        UniformSegmentPartitioner(context).partition(_snapshot())


def test_snapshot_without_audio_is_refused() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hi"),)),
    ))
    with pytest.raises(ValueError, match="AudioPart"):
        UniformSegmentPartitioner().partition(snapshot)


def test_explain_runs_with_the_registered_partition() -> None:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock_audio", "model_id": "stub"},
        "explanation": {
            "players": "uniform_segments",
            "absence_policy": "delete",
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config).explain_sync(_snapshot())
    assert len(output.result.values) == 3
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value
    )
