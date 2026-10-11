"""Known-truth CTC model and synthetic speech shared by the alignment tests."""

from collections.abc import Mapping

import numpy as np

from mllm_shap.alignment.io import encode_wav

RATE = 8000
FRAME = 400  # samples per CTC frame (50 ms)
VOCAB = {"<pad>": 0, "A": 1, "B": 2, "C": 3, "|": 4}
# Frame labels: silence, "A" "B" burst, silence with the word separator "|",
# "C" burst, silence.
FRAMES = [0, 0, 1, 1, 2, 2, 0, 4, 0, 3, 3, 3, 0, 0]


class FakeCtc:
    """CTC model whose log-probabilities follow ``FRAMES``."""

    sample_rate = RATE
    blank_id = 0
    vocab: Mapping[str, int] = VOCAB

    def log_probs(self, waveform: np.ndarray) -> np.ndarray:
        assert len(waveform) == len(FRAMES) * FRAME
        lp = np.full((len(FRAMES), len(VOCAB)), -20.0)
        lp[np.arange(len(FRAMES)), FRAMES] = 0.0
        return lp


def speech() -> bytes:
    """Tone where ``FRAMES`` has a letter, silence elsewhere."""
    t = np.arange(FRAME) / RATE
    tone = 0.6 * np.sin(2 * np.pi * 300 * t)
    chunks = [tone if label in {1, 2, 3} else np.zeros(FRAME) for label in FRAMES]
    return encode_wav(np.concatenate(chunks), RATE)
