"""WAV encoding and decoding to mono float waveforms.

@see [[src/nlp_shap/alignment/io.py]]
"""

import io

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from nlp_shap.alignment.io import decode_audio, encode_wav  # noqa: E402


def _tone(seconds: float = 0.1, rate: int = 8000) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_wav_round_trip_keeps_samples_within_pcm16_precision() -> None:
    wave = _tone()
    decoded, rate = decode_audio(encode_wav(wave, 8000))
    assert rate == 8000
    assert decoded.dtype == np.float32
    assert np.max(np.abs(decoded - wave)) < 1 / 32767 + 1e-6


def test_decode_averages_channels_to_mono() -> None:
    left, right = _tone(), np.zeros(800, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([left, right], axis=1), 8000, format="WAV")
    decoded, _ = decode_audio(buffer.getvalue())
    assert decoded.ndim == 1
    assert np.max(np.abs(decoded - left / 2)) < 2 / 32767


def test_decode_resamples_to_the_requested_rate() -> None:
    pytest.importorskip("librosa")
    decoded, rate = decode_audio(encode_wav(_tone(), 8000), sample_rate=16000)
    assert rate == 16000
    assert decoded.shape == (1600,)


def test_encode_clips_out_of_range_samples() -> None:
    decoded, _ = decode_audio(encode_wav(np.array([2.0, -2.0, np.nan]), 8000))
    assert decoded.tolist() == pytest.approx([1.0, -1.0, 0.0], abs=1e-4)


def test_decode_rejects_undecodable_bytes() -> None:
    with pytest.raises(ValueError, match="decode"):
        decode_audio(b"not audio")
