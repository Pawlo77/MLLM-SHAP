"""Audio players from SGPA segments, rendered by every absence policy.

@see [[src/mllm_shap/alignment/partitions.py]]
@see [[src/mllm_shap/backends/mock/audio.py]]
"""

import asyncio

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("soundfile")

from mllm_shap import ExplainConfig, ExplainRunner, PluginGroup
from mllm_shap.alignment.io import decode_audio
from mllm_shap.alignment.partitions import SgpaSegmentPartitioner
from mllm_shap.alignment.sgpa import SpectrogramGuidedAligner
from mllm_shap.backends.mock import MockAudioBackend
from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.masking.policies import (
    DeletePolicy,
    NeutralPolicy,
    PadPolicy,
)
from mllm_shap.plugins import default_registry
from tests.alignment._fakes import FRAME, FRAMES, RATE, FakeCtc, speech


def _snapshot(transcript: str = "ab — c") -> ConversationSnapshot:
    spoken = Message(
        role=Role.USER, content=(TextPart(transcript), AudioPart(speech(), RATE))
    )
    question = Message(role=Role.USER, text="What was said?")
    return ConversationSnapshot.from_turns((Turn(messages=(question, spoken)),))


def _partitioner() -> SgpaSegmentPartitioner:
    return SgpaSegmentPartitioner(aligner=SpectrogramGuidedAligner(FakeCtc()))


def _audio(snapshot: ConversationSnapshot) -> np.ndarray:
    part = snapshot.turns[0].messages[1].content[1]
    assert isinstance(part, AudioPart)
    return decode_audio(part.data())[0]


def test_players_are_spellable_words_with_text_and_sample_spans() -> None:
    snapshot = _snapshot()
    players = _partitioner().partition(snapshot)
    sid = snapshot.snapshot_id
    assert players.player_ids == (f"{sid}:0:1:0", f"{sid}:0:1:2")
    first, second = players.spans
    assert (first.message_index, first.start, first.end, first.text) == (1, 0, 2, "ab")
    assert (second.start, second.end, second.text) == (5, 6, "c")
    assert first.samples is not None
    assert second.samples is not None
    assert first.samples[1] == second.samples[0]
    assert second.samples[1] <= len(FRAMES) * FRAME


def test_delete_cuts_absent_audio_and_text() -> None:
    snapshot = _snapshot()
    players = _partitioner().partition(snapshot)
    rendered = DeletePolicy().apply(snapshot, players, CoalitionMask((True, False)))
    assert rendered.turns[0].messages[1].text == "ab — "  # the dash is no player
    start, end = players.spans[1].samples or (0, 0)
    assert len(_audio(rendered)) == len(_audio(snapshot)) - (end - start)


@pytest.mark.parametrize("policy", [PadPolicy(), NeutralPolicy()])
def test_pad_and_neutral_silence_absent_audio(
    policy: PadPolicy | NeutralPolicy,
) -> None:
    snapshot = _snapshot()
    players = _partitioner().partition(snapshot)
    rendered = policy.apply(snapshot, players, CoalitionMask((False, True)))
    before, after = _audio(snapshot), _audio(rendered)
    start, end = players.spans[0].samples or (0, 0)
    assert len(after) == len(before)
    assert np.max(np.abs(after[start:end])) == 0.0
    assert np.array_equal(after[end:], before[end:])


def test_snapshot_without_audio_is_refused() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hi"),)),
    ))
    with pytest.raises(ValueError, match="AudioPart"):
        _partitioner().partition(snapshot)


def test_audio_without_transcript_is_refused() -> None:
    silent = Message(role=Role.USER, content=(AudioPart(speech(), RATE),))
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(silent,)),))
    with pytest.raises(ValueError, match="transcript"):
        _partitioner().partition(snapshot)


@pytest.mark.parametrize("backend", ["mock", "mock_audio"])
def test_explain_runs_with_audio_players_end_to_end(backend: str) -> None:
    """Both mock backends score the rendered coalitions; efficiency holds."""
    registry = default_registry()
    registry.register(
        PluginGroup.PARTITIONS,
        "sgpa_segments",
        lambda context: _partitioner(),
        replace=True,
    )
    config = ExplainConfig.model_validate({
        "backend": {"kind": backend, "model_id": "stub"},
        "explanation": {
            "players": "sgpa_segments",
            "absence_policy": "delete",
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config, registry=registry).explain_sync(_snapshot())
    assert output.player_set is not None
    assert [span.text for span in output.player_set.spans] == ["ab", "c"]
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value
    )


def test_mock_audio_hears_only_present_audio_players() -> None:
    """Silencing a word's samples drops exactly its heard token."""
    snapshot = _snapshot()
    players = _partitioner().partition(snapshot)
    backend = MockAudioBackend(model_id="stub")

    def heard(present: tuple[bool, bool]) -> list[str]:
        rendered = PadPolicy().apply(snapshot, players, CoalitionMask(present))
        text = asyncio.run(backend.generate(rendered, GenerationConfig())).text
        return text.split()[-2:]

    both, first, neither = (
        heard((True, True)),
        heard((True, False)),
        heard((False, False)),
    )
    assert both[0] != both[1]
    assert first[-1] == both[0]
    assert both[0] not in neither
    assert both[1] not in neither + first


def test_options_reach_the_default_aligner() -> None:
    """``plugin_options.sgpa_segments`` configures the Wav2Vec2 model lazily."""
    from mllm_shap.alignment.partitions import SgpaOptions

    options = SgpaOptions(model_id="m", revision="r", device="cpu", flux_weight=0.5)
    assert (options.model_id, options.revision, options.flux_weight) == ("m", "r", 0.5)
    with pytest.raises(ValueError, match="greater than or equal"):
        SgpaOptions(energy_weight=-1.0)


def test_text_roles_add_word_players_of_text_messages() -> None:
    """Typed and spoken words are players of one run; only spoken ones have samples."""
    from mllm_shap.plugins.context import PluginContext

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"players": "sgpa_segments"},
    })
    context = PluginContext(config=config, options={"text_roles": ["user"]})
    snapshot = _snapshot()
    players = SgpaSegmentPartitioner(
        context, aligner=SpectrogramGuidedAligner(FakeCtc())
    ).partition(snapshot)
    sid = snapshot.snapshot_id
    assert [span.text for span in players.spans] == ["What", "was", "said?", "ab", "c"]
    assert [span.samples is None for span in players.spans] == [True] * 3 + [False] * 2
    assert players.player_ids[:3] == (f"{sid}:0:0:0", f"{sid}:0:0:1", f"{sid}:0:0:2")


def test_text_roles_default_to_audio_players_only() -> None:
    players = _partitioner().partition(_snapshot())
    assert all(span.samples is not None for span in players.spans)


def _align_only() -> SgpaSegmentPartitioner:
    from mllm_shap.plugins.context import PluginContext

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock_audio", "model_id": "stub"},
        "explanation": {"players": "sgpa_segments"},
    })
    context = PluginContext(config=config, options={"transcript": "align_only"})
    return SgpaSegmentPartitioner(context, aligner=SpectrogramGuidedAligner(FakeCtc()))


@pytest.mark.parametrize("policy", [DeletePolicy(), PadPolicy(), NeutralPolicy()])
def test_align_only_transcript_never_reaches_the_model(
    policy: DeletePolicy | PadPolicy | NeutralPolicy,
) -> None:
    """The transcript builds the players; every coalition sends audio only."""
    snapshot = _snapshot()
    players = _align_only().partition(snapshot)
    assert players.hidden_text == (1,)
    for present in ((True, True), (True, False), (False, False)):
        rendered = policy.apply(snapshot, players, CoalitionMask(present))
        question, spoken = rendered.turns[0].messages
        assert question.text == "What was said?"
        assert [type(part) for part in spoken.content] == [AudioPart]


def test_align_only_values_come_from_the_audio_alone() -> None:
    """With mock_audio each spoken word moves the answer only through its sound."""
    registry = default_registry()
    registry.register(
        PluginGroup.PARTITIONS,
        "sgpa_segments",
        lambda context: _align_only(),
        replace=True,
    )
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock_audio", "model_id": "stub"},
        "explanation": {
            "players": "sgpa_segments",
            "absence_policy": "pad",
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config, registry=registry).explain_sync(_snapshot())
    result = output.result
    assert all(value > 0.0 for value in result.values)
    assert sum(result.values) == pytest.approx(result.grand_value - result.base_value)
