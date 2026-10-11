"""Audio segment figures: waveform, boundaries, CTC shifts and energy.

@see [[src/mllm_shap/viz/audio.py]]
"""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("librosa")
pytest.importorskip("soundfile")

from mllm_shap.alignment.sgpa import SpectrogramGuidedAligner
from mllm_shap.domain.conversation import AudioPart
from mllm_shap.viz import plot_audio_segments
from tests.alignment._fakes import RATE, FakeCtc, speech

AUDIO = AudioPart(speech(), RATE)


def _segments() -> list:
    return SpectrogramGuidedAligner(FakeCtc()).align("ab c", AUDIO)


def test_segments_record_their_rough_ctc_edges() -> None:
    first, second = _segments()
    assert (first.ctc_start_time, first.ctc_end_time) == pytest.approx((0.1, 0.3))
    assert (second.ctc_start_time, second.ctc_end_time) == pytest.approx((0.45, 0.6))


def test_figure_shows_words_boundaries_shifts_and_energy() -> None:
    segments = _segments()
    figure = plot_audio_segments(AUDIO, segments)
    wave, energy = figure.axes
    assert energy.get_xlabel() == "time (s)"
    assert [t.get_text() for t in wave.texts if t.get_text()] == ["ab", "c"]
    refined = sorted({s.start_time for s in segments} | {s.end_time for s in segments})
    lines = sorted(
        line.get_xdata()[0]
        for line in wave.get_lines()
        if line.get_linestyle() == "-" and line.get_label() == "boundary"
    )
    assert lines == pytest.approx(refined)
    assert len([p for p in wave.patches if p.get_label() == "segment"]) == 2
    assert wave.get_legend() is not None


def test_values_colour_segments_by_sign() -> None:
    figure = plot_audio_segments(AUDIO, _segments(), values=[0.4, -0.2])
    first, second = (p for p in figure.axes[0].patches if p.get_label() == "segment")
    red, blue = first.get_facecolor(), second.get_facecolor()
    assert red[0] > red[2]
    assert blue[2] > blue[0]


def test_values_must_match_the_segments() -> None:
    with pytest.raises(ValueError, match="one value per segment"):
        plot_audio_segments(AUDIO, _segments(), values=[1.0])
