"""Spectrogram-guided alignment with a fake CTC model over synthetic tone bursts.

@see [[src/mllm_shap/alignment/sgpa.py]]
"""

import numpy as np
import pytest

pytest.importorskip("librosa")
pytest.importorskip("soundfile")

from mllm_shap.alignment.io import decode_audio, encode_wav
from mllm_shap.alignment.sgpa import SpectrogramGuidedAligner
from mllm_shap.domain.conversation import AudioPart
from tests.alignment._fakes import FRAME, FRAMES, RATE, FakeCtc, speech


def _align(transcript: str, **kwargs: bool) -> list:
    aligner = SpectrogramGuidedAligner(FakeCtc())
    return aligner.align(transcript, AudioPart(speech(), RATE), **kwargs)


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
    segments = _align("ab — c")
    assert [(s.token, s.token_index) for s in segments] == [("ab", 0), ("c", 2)]


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
        SpectrogramGuidedAligner(FakeCtc(), energy_weight=0.0, flux_weight=0.0)


@pytest.mark.gpu
def test_wav2vec2_log_probs_are_normalised_per_frame() -> None:
    """A tiny random Wav2Vec2 checkpoint loads and yields log-probabilities."""
    pytest.importorskip("transformers")
    from mllm_shap.alignment.sgpa import Wav2Vec2Ctc

    model = Wav2Vec2Ctc("hf-internal-testing/tiny-random-wav2vec2", revision="main")
    lp = model.log_probs(np.zeros(model.sample_rate, dtype=np.float32))
    assert lp.ndim == 2
    assert lp.shape[1] == len(model.vocab)
    assert np.exp(lp).sum(axis=1) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.gpu
def test_wav2vec2_load_prints_no_report(capfd: pytest.CaptureFixture[str]) -> None:
    """The expected missing ``masked_spec_embed`` does not print a load report."""
    pytest.importorskip("transformers")
    from mllm_shap.alignment.sgpa import Wav2Vec2Ctc

    Wav2Vec2Ctc("hf-internal-testing/tiny-random-wav2vec2", revision="main")
    assert "LOAD REPORT" not in capfd.readouterr().err


def test_loud_audio_keeps_unrefined_boundaries() -> None:
    """Without a quiet point the boundary stays at the middle of the CTC gap."""
    noise = np.random.default_rng(0).uniform(-0.6, 0.6, len(FRAMES) * FRAME)
    audio = AudioPart(encode_wav(noise, RATE), RATE)
    segments = SpectrogramGuidedAligner(FakeCtc()).align("ab c", audio)
    assert [s.boundary_refined for s in segments] == [False, False]
    assert 0.3 < segments[0].end_time == segments[1].start_time < 0.45


def test_short_search_window_keeps_the_rough_edge() -> None:
    """A gap narrower than one analysis frame cannot be refined."""
    aligner = SpectrogramGuidedAligner(FakeCtc())
    waveform = np.zeros(100, dtype=np.float32)
    assert aligner._refine(waveform, RATE, 0.005, 0.001) == (0.005, False)
