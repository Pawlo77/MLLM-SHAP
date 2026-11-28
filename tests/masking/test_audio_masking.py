"""Tests for rendering audio players.

@see [[src/mllm_shap/masking/audio.py]]
"""

import numpy as np
import pytest

pytest.importorskip("soundfile")

from mllm_shap.alignment.io import decode_audio, encode_wav
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.domain.players import PlayerSpan
from mllm_shap.masking.audio import render_audio

RATE = 8_000
"""Sample rate of the test clips."""


def test_render_audio_edits_the_clip_its_span_names() -> None:
    first = AudioPart(encode_wav(np.full(800, 0.5), RATE), RATE)
    second = AudioPart(encode_wav(np.full(400, 0.5), RATE), RATE)
    message = Message(role=Role.USER, content=(TextPart("x"), first, second))
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(message,)),))
    span = PlayerSpan(
        message_index=0, start=0, end=1, text="x", samples=(0, 200), part=1
    )
    rendered = render_audio(snapshot, [span], [False], cut=True).messages()[0]
    kept_first, kept_second = rendered.content[1:]
    assert kept_first == first
    assert isinstance(kept_second, AudioPart)
    assert len(decode_audio(kept_second.data())[0]) == 200
