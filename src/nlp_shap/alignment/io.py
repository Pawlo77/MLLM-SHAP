"""Encoded audio bytes to mono float waveforms and back.

@domain alignment
@see [[knowledge/architecture/alignment]]
@tested-by [[tests/alignment/test_io.py]]
"""

import importlib
import io
from types import ModuleType

import numpy as np
from numpy.typing import NDArray


def require(module: str) -> ModuleType:
    """Import an optional dependency of the ``audio`` or ``transformers`` extra.

    Raises:
        ImportError: Naming the extra to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extra = "transformers" if module in {"torch", "transformers"} else "audio"
        msg = f"{module} is required for nlp_shap.alignment; install nlp-shap[{extra}]"
        raise ImportError(msg) from exc


def decode_audio(
    data: bytes, sample_rate: int | None = None
) -> tuple[NDArray[np.float32], int]:
    """Decode ``data`` to a mono float32 waveform, resampled to ``sample_rate``.

    Channels are averaged. Without ``sample_rate`` the file's own rate is kept.

    Raises:
        ValueError: If ``data`` is not a format ``soundfile`` can read.
    """
    soundfile = require("soundfile")
    try:
        frames, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except (RuntimeError, soundfile.LibsndfileError) as exc:
        msg = "could not decode the audio bytes"
        raise ValueError(msg) from exc
    waveform = frames.mean(axis=1).astype(np.float32)
    if sample_rate is not None and sample_rate != rate:
        waveform = require("librosa").resample(
            waveform, orig_sr=rate, target_sr=sample_rate
        )
        rate = sample_rate
    return waveform.astype(np.float32), int(rate)


def encode_wav(waveform: NDArray[np.floating], sample_rate: int) -> bytes:
    """Encode a mono waveform as 16-bit PCM WAV; NaN becomes 0, values clip to ±1."""
    soundfile = require("soundfile")
    samples = np.clip(np.nan_to_num(np.asarray(waveform, dtype=np.float32)), -1, 1)
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
