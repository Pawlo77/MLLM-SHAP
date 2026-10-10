"""Spectrogram-guided alignment with a fake CTC model over synthetic tone bursts.

@see [[src/nlp_shap/alignment/sgpa.py]]
"""

from collections.abc import Mapping

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("soundfile")

from nlp_shap.alignment.io import decode_audio, encode_wav
from nlp_shap.alignment.sgpa import SpectrogramGuidedAligner
from nlp_shap.domain.conversation import AudioPart

RATE = 8000
FRAME = 400  # samples per CTC frame (50 ms)
VOCAB = {"<pad>": 0, "A": 1, "B": 2, "C": 3, "|": 4}
# Frame labels: silence, "A" "B" burst, silence, "C" burst, silence.
FRAMES = [0, 0, 1, 1, 2, 2, 0, 0, 0, 3, 3, 3, 0, 0]


class _FakeCtc:
    """Known-truth CTC model: log-probabilities follow ``FRAMES``."""

    sample_rate = RATE
    blank_id = 0
    vocab: Mapping[str, int] = VOCAB

    def log_probs(self, waveform: np.ndarray) -> np.ndarray:
        assert len(waveform) == len(FRAMES) * FRAME
        lp = np.full((len(FRAMES), len(VOCAB)), -20.0)
        lp[np.arange(len(FRAMES)), FRAMES] = 0.0
        return lp


def _speech() -> bytes:
    """Tone where ``FRAMES`` has a letter, silence elsewhere."""
    t = np.arange(FRAME) / RATE
    tone = 0.6 * np.sin(2 * np.pi * 300 * t)
    chunks = [tone if label else np.zeros(FRAME) for label in FRAMES]
    return encode_wav(np.concatenate(chunks), RATE)


def _align(transcript: str, **kwargs: bool) -> list:
    aligner = SpectrogramGuidedAligner(_FakeCtc())
    return aligner.align(transcript, AudioPart(_speech(), RATE), **kwargs)


def test_tokens_align_to_their_bursts_with_a_shared_refined_boundary() -> None:
    first, second = _align("ab c")
    assert (first.token, second.token) == ("ab", "c")
    assert first.start_time <= 0.1 + 1e-9
    assert 0.3 <= first.end_time == second.start_time <= 0.45
    assert second.end_time >= 0.6
    assert first.boundary_refined
    assert first.confidence == pytest.approx(1.0)
    assert (first.start_sample, first.end_sample) == (
        round(first.start_time * RATE),
        round(first.end_time * RATE),
    )


def test_tokens_without_model_characters_are_skipped() -> None:
    assert [segment.token for segment in _align("ab — c")] == ["ab", "c"]


def test_attached_audio_is_the_segment_slice() -> None:
    segment = _align("ab c", attach_audio=True)[0]
    waveform, rate = decode_audio(segment.audio)
    assert rate == RATE
    assert len(waveform) == segment.end_sample - segment.start_sample


def test_transcript_without_model_characters_is_refused() -> None:
    with pytest.raises(ValueError, match="no characters"):
        _align("— …")


def test_boundary_weights_must_be_usable() -> None:
    with pytest.raises(ValueError, match="weight"):
        SpectrogramGuidedAligner(_FakeCtc(), energy_weight=0.0, flux_weight=0.0)


@pytest.mark.gpu
def test_wav2vec2_log_probs_are_normalised_per_frame() -> None:
    """A tiny random Wav2Vec2 checkpoint loads and yields log-probabilities."""
    pytest.importorskip("transformers")
    from nlp_shap.alignment.sgpa import Wav2Vec2Ctc

    model = Wav2Vec2Ctc("hf-internal-testing/tiny-random-wav2vec2", revision="main")
    lp = model.log_probs(np.zeros(model.sample_rate, dtype=np.float32))
    assert lp.ndim == 2
    assert lp.shape[1] == len(model.vocab)
    assert np.exp(lp).sum(axis=1) == pytest.approx(1.0, abs=1e-4)
