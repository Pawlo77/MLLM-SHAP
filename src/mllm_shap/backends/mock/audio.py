"""Deterministic mock backend that hears the voiced runs of audio parts.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_mock_audio.py]]
"""

import hashlib
import io
import wave

import numpy as np
from numpy.typing import NDArray

from ...alignment.io import decode_audio
from ...domain.config import GenerationConfig
from ...domain.conversation import AudioPart, ConversationSnapshot, TextPart
from ...domain.generation import GenerationRecord
from ...value.generation_records import generation_record_from_text
from .._chat import message_chat_text
from .backend import MockBackend

SILENCE = 0.01
"""Largest absolute amplitude (of full scale 1) that counts as silence."""

MIN_GAP_S = 0.01
"""Shortest silence that separates two voiced runs; shorter dips (zero crossings)
stay inside one run."""


class MockAudioBackend(MockBackend):
    """Echo text words and answer one token per voiced run of every audio part.

    A run's token hashes its 16-bit samples, so silencing or cutting a run drops
    exactly its token and leaves the others unchanged. 16-bit PCM WAV decodes
    with the standard library; other formats need the ``audio`` extra.
    """

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        """Return the heard words of ``snapshot``, at most ``max_new_tokens``."""
        words: list[str] = []
        for message in snapshot.messages():
            words.extend(message_chat_text(message).split())
            for part in message.content:
                if not isinstance(part, TextPart):
                    words.extend(_run_tokens(*_waveform(part)))
        return generation_record_from_text(" ".join(words[: generation.max_new_tokens]))


def _waveform(part: AudioPart) -> tuple[NDArray[np.float32], int]:
    data = part.data()
    try:
        with wave.open(io.BytesIO(data)) as reader:
            width, channels = reader.getsampwidth(), reader.getnchannels()
            rate = reader.getframerate()
            frames = reader.readframes(reader.getnframes())
    except (wave.Error, EOFError):
        return decode_audio(data)
    if width != 2:
        return decode_audio(data)
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    return (samples.mean(axis=1) / 32768).astype(np.float32), rate


def _run_tokens(waveform: NDArray[np.float32], rate: int) -> list[str]:
    loud = np.flatnonzero(np.abs(waveform) > SILENCE)
    if loud.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(loud) > max(1, int(MIN_GAP_S * rate)))
    starts = np.r_[loud[0], loud[breaks + 1]]
    ends = np.r_[loud[breaks], loud[-1]] + 1
    tokens: list[str] = []
    for start, end in zip(starts, ends, strict=True):
        scaled = np.clip(np.round(waveform[start:end] * 32768), -32768, 32767)
        pcm = scaled.astype("<i2").tobytes()
        tokens.append(hashlib.sha256(pcm).hexdigest()[:4])
    return tokens
