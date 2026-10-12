"""Tests for the mock backend that hears audio parts; no audio extra needed.

@see [[src/mllm_shap/backends/mock/audio.py]]
"""

import asyncio
import io
import wave

import numpy as np
import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.backends.mock import MockAudioBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role

RATE = 8000
CHUNK = 400  # 50 ms


def _wav(chunks: list[float]) -> bytes:
    """Encode 50 ms chunks: a 300 Hz tone of the given amplitude, 0 for silence."""
    t = np.arange(CHUNK) / RATE
    waveform = np.concatenate([a * np.sin(2 * np.pi * 300 * t) for a in chunks])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes((waveform * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _snapshot(audio: bytes, text: str = "say it") -> ConversationSnapshot:
    message = Message(role=Role.USER, content=(TextPart(text), AudioPart(audio, RATE)))
    return ConversationSnapshot.from_turns((Turn(messages=(message,)),))


def _heard(snapshot: ConversationSnapshot, max_new_tokens: int = 16) -> list[str]:
    record = asyncio.run(
        MockAudioBackend(model_id="stub").generate(
            snapshot, GenerationConfig(max_new_tokens=max_new_tokens)
        )
    )
    return record.text.split()


def test_text_words_are_echoed_and_truncated() -> None:
    message = Message(role=Role.USER, text="one two three", speaker="ann")
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(message,)),))
    assert _heard(snapshot) == ["ann:", "one", "two", "three"]
    assert _heard(snapshot, max_new_tokens=2) == ["ann:", "one"]


def test_each_voiced_run_adds_one_token() -> None:
    words = _heard(_snapshot(_wav([0, 0.6, 0.6, 0, 0.4, 0])))
    assert words[:2] == ["say", "it"]
    assert len(words) == 4
    assert words[2] != words[3]


def test_silencing_or_cutting_a_run_drops_only_its_token() -> None:
    full = _heard(_snapshot(_wav([0.6, 0, 0.4])))
    silenced = _heard(_snapshot(_wav([0.6, 0, 0])))
    cut = _heard(_snapshot(_wav([0.6, 0])))
    assert silenced == cut == full[:3]


def test_output_is_deterministic() -> None:
    snapshot = _snapshot(_wav([0.6, 0, 0.4]))
    assert _heard(snapshot) == _heard(snapshot)


def test_undecodable_audio_is_refused() -> None:
    with pytest.raises((ValueError, ImportError)):
        _heard(_snapshot(b"not audio"))


def test_explain_runs_on_a_multimodal_snapshot() -> None:
    """Text players of a message with audio; efficiency holds without the extra."""
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock_audio", "model_id": "stub"},
        "explanation": {"archive": {"path": ""}},
    })
    output = ExplainRunner(config).explain_sync(_snapshot(_wav([0.6, 0, 0.4])))
    result = output.result
    assert len(result.values) == 2
    assert sum(result.values) == pytest.approx(result.grand_value - result.base_value)


def test_silent_audio_adds_no_token() -> None:
    assert _heard(_snapshot(_wav([0, 0]))) == ["say", "it"]


def test_non_16_bit_wav_goes_through_the_audio_extra() -> None:
    """8-bit WAV uses ``decode_audio``; its runs are heard like 16-bit ones."""
    pytest.importorskip("soundfile")
    t = np.arange(CHUNK) / RATE
    burst = 0.6 * np.sin(2 * np.pi * 300 * t)
    waveform = np.concatenate([burst, np.zeros(CHUNK), burst])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(1)
        writer.setframerate(RATE)
        writer.writeframes((waveform * 127 + 128).astype(np.uint8).tobytes())
    assert len(_heard(_snapshot(buffer.getvalue()))) == 4
