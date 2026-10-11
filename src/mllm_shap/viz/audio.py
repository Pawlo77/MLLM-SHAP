"""Figure of aligned audio segments: waveform, refined boundaries and energy.

@domain viz
@see [[knowledge/architecture/viz]]
@tested-by [[tests/viz/test_audio.py]]
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..alignment.io import decode_audio
from ..alignment.segments import AudioSegment
from ..domain.conversation import AudioPart
from .colors import diverging_rgba
from .style import TEXT_MUTED, _import_pyplot, apply_attribution_theme

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_FRAME, _HOP = 256, 64  # RMS window and hop in samples, as in the aligner
_SHADES = ("#E5E7EB", "#F3F4F6")  # alternating segment fills without values
_MAX_ALPHA = 0.55  # value fills stay translucent so the waveform shows through
_LIGHTEN = 0.35  # every second value fill moves this far toward white
_BOUNDARY, _CTC = "#111827", "#6B7280"
_MIN_SHIFT_S = 0.005  # shorter moves get no arrow


def plot_audio_segments(
    audio: AudioPart,
    segments: Sequence[AudioSegment],
    values: Sequence[float] | None = None,
) -> "Figure":
    """Draw ``audio`` with its segments and how refinement moved each boundary.

    The top panel shades each segment (by the sign and size of ``values`` when
    given, as attribution renderers do), labels its token, draws the refined
    boundaries as solid lines and each rough CTC edge as a dashed line with an
    arrow to where refinement moved it. The bottom panel shows the RMS energy
    the refinement looks for minima of. Needs the ``viz`` and ``audio`` extras.

    Raises:
        ValueError: If ``values`` does not hold one value per segment.
    """
    if values is not None and len(values) != len(segments):
        msg = "values must hold one value per segment"
        raise ValueError(msg)
    waveform, rate = decode_audio(audio.data())
    plt = _import_pyplot()
    apply_attribution_theme()
    figure, (wave, energy) = cast(Any, plt).subplots(
        2, 1, figsize=(10, 4.2), sharex=True, height_ratios=(3, 1)
    )
    seconds = np.arange(len(waveform)) / rate
    wave.plot(seconds, waveform, color=TEXT_MUTED, linewidth=0.6, alpha=0.7)
    _draw_segments(wave, segments, values)
    _draw_energy(energy, waveform, rate, segments)
    handles, labels = wave.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    wave.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8)
    wave.set_ylabel("amplitude")
    figure.tight_layout()
    return cast("Figure", figure)


def _draw_segments(
    axis: Any, segments: Sequence[AudioSegment], values: Sequence[float] | None
) -> None:
    vmax = max((abs(v) for v in values or ()), default=0.0)
    boundaries: set[float] = set()
    for index, segment in enumerate(segments):
        fill = (
            _value_fill(values[index], vmax, lighten=index % 2 == 1)
            if values is not None
            else _SHADES[index % 2]
        )
        axis.axvspan(segment.start_time, segment.end_time, color=fill, label="segment")
        axis.text(
            (segment.start_time + segment.end_time) / 2,
            1.01,
            segment.token,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        boundaries |= {segment.start_time, segment.end_time}
        for rough, refined in (
            (segment.ctc_start_time, segment.start_time),
            (segment.ctc_end_time, segment.end_time),
        ):
            _draw_shift(axis, rough, refined)
    for time in sorted(boundaries):
        axis.axvline(time, color=_BOUNDARY, linewidth=1.1, label="boundary")


def _value_fill(
    value: float, vmax: float, lighten: bool
) -> tuple[float, float, float, float]:
    """Attribution colour, translucent; alternate neighbours are lighter."""
    red, green, blue, alpha = diverging_rgba(value, vmax)
    if lighten:
        red, green, blue = (c + (1.0 - c) * _LIGHTEN for c in (red, green, blue))
    return red, green, blue, min(alpha, _MAX_ALPHA)


def _draw_shift(axis: Any, rough: float, refined: float) -> None:
    axis.axvline(rough, color=_CTC, linewidth=0.8, linestyle="--", label="CTC edge")
    if abs(refined - rough) >= _MIN_SHIFT_S:
        axis.annotate(
            "",
            xy=(refined, 0.9),
            xytext=(rough, 0.9),
            xycoords=axis.get_xaxis_transform(),
            arrowprops={"arrowstyle": "->", "color": _CTC, "linewidth": 0.8},
        )


def _draw_energy(
    axis: Any,
    waveform: np.ndarray,
    rate: int,
    segments: Sequence[AudioSegment],
) -> None:
    starts = np.arange(0, max(len(waveform) - _FRAME, 0) + 1, _HOP)
    rms = np.array([np.sqrt(np.mean(waveform[s : s + _FRAME] ** 2)) for s in starts])
    axis.plot((starts + _FRAME / 2) / rate, rms, color=_BOUNDARY, linewidth=0.9)
    for time in sorted(
        {s.start_time for s in segments} | {s.end_time for s in segments}
    ):
        axis.axvline(time, color=_BOUNDARY, linewidth=1.1, alpha=0.5)
    axis.set_ylabel("RMS")
    axis.set_xlabel("time (s)")
