"""Performance benchmark for the audio mock backend.

@see [[src/mllm_shap/backends/mock/audio.py]]
"""

import asyncio
import io
import wave

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

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

RATE = 16_000


def _speech(seconds: int) -> bytes:
    # Alternating 100 ms tone bursts and pauses: five voiced runs per second.
    t = np.arange(RATE // 10) / RATE
    burst = 0.5 * np.sin(2 * np.pi * 220 * t)
    pause = np.zeros_like(burst)
    waveform = np.tile(np.concatenate([burst, pause]), 5 * seconds)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(RATE)
        writer.writeframes((waveform * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.mark.bench
def test_mock_audio_generate_ten_seconds(benchmark: BenchmarkFixture) -> None:
    """One generate call over ten seconds of 16 kHz speech with 50 voiced runs."""
    message = Message(
        role=Role.USER, content=(TextPart("describe"), AudioPart(_speech(10), RATE))
    )
    snapshot = ConversationSnapshot.from_turns((Turn(messages=(message,)),))
    backend = MockAudioBackend(model_id="bench")
    generation = GenerationConfig(max_new_tokens=256)

    record = benchmark(lambda: asyncio.run(backend.generate(snapshot, generation)))

    assert len(record.text.split()) == 51
