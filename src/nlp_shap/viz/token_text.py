"""SHAP-style inline colored token attribution renderer.

@domain viz
@see [[knowledge/architecture/viz]]
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.estimands import Estimand
from ..protocols.plugin import PluginOptions
from .colors import token_background_css, token_background_rgba
from .labels import check_attribution
from .style import new_attribution_figure, polish_text_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..plugins.context import PluginContext


_BOX_PAD = 0.35
"""Token box padding, as a fraction of the font size."""
_HEADER_INCHES = 1.4
"""Figure height outside the token lines (title, legend, margins)."""
_LINE_INCHES = 0.5
"""Figure height of one line of tokens."""


class TokenTextRenderer:
    """Render attributions as inline colored tokens."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def name(self) -> str:
        """Return the registered renderer identifier."""
        return "token_text"

    def render(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> "Figure":
        """Build a matplotlib figure with SHAP-style colored token labels."""
        check_attribution(labels, values)

        vmax = max(abs(value) for value in values)
        header = title or f"{estimand.value.title()} token attributions"
        width = max(7.5, min(14.0, 1.1 + sum(len(label) for label in labels) * 0.11))
        fig, axis = new_attribution_figure(width=width, height=2.4)
        polish_text_axes(axis)
        axis.set_title(
            header,
            loc="left",
            fontsize=13,
            fontweight="bold",
            color="#111827",
            pad=12,
        )
        fontsize = 12.0
        tokens = [
            axis.text(
                0.0,
                0.0,
                label,
                transform=axis.transAxes,
                fontsize=fontsize,
                va="center",
                ha="left",
                color="#111827",
                bbox={
                    "boxstyle": f"round,pad={_BOX_PAD},rounding_size=0.08",
                    "facecolor": token_background_rgba(value, vmax),
                    "edgecolor": "none",
                },
            )
            for label, value in zip(labels, values, strict=True)
        ]
        fig.tight_layout()
        lines = _wrap_tokens(axis, tokens, fontsize)
        fig.set_size_inches(width, _HEADER_INCHES + _LINE_INCHES * len(lines))
        fig.tight_layout()
        _place_tokens(axis, tokens, lines, fontsize)
        _draw_color_legend(axis)
        return fig

    def to_html(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> str:
        """Return an HTML fragment with colored token spans for Jupyter."""
        check_attribution(labels, values)
        vmax = max(abs(value) for value in values)
        header = title or f"{estimand.value.title()} token attributions"
        spans: list[str] = []
        for label, value in zip(labels, values, strict=True):
            background = token_background_css(value, vmax)
            value_label = f"{value:+.3f}"
            spans.append(
                '<span class="nlp-shap-token" '
                f'title="{estimand.value}: {value_label}" '
                f'style="background-color:{background};">'
                f"{_escape_html(label)}</span>"
            )
        body = "".join(spans)
        return (
            '<div class="nlp-shap-attribution" '
            'style="font-family:system-ui,-apple-system,Segoe UI,Roboto,'
            'Helvetica Neue,Arial,sans-serif;line-height:1.8;color:#111827;">'
            f'<div style="font-size:14px;font-weight:600;margin-bottom:8px;">'
            f"{_escape_html(header)}</div>"
            '<div style="font-size:12px;color:#6B7280;margin-bottom:10px;">'
            '<span style="display:inline-block;width:10px;height:10px;'
            'border-radius:2px;background:#FF0D57;margin-right:4px;"></span>'
            "increases&nbsp;&nbsp;"
            '<span style="display:inline-block;width:10px;height:10px;'
            'border-radius:2px;background:#1E88E5;margin-right:4px;"></span>'
            "decreases</div>"
            f'<div style="font-size:15px;word-break:break-word;">{body}</div>'
            "</div>"
            "<style>"
            ".nlp-shap-token{display:inline-block;padding:3px 7px;margin:2px 3px 2px 0;"
            "border-radius:4px;transition:box-shadow .15s ease;}"
            ".nlp-shap-token:hover{box-shadow:0 0 0 1px rgba(17,24,39,.12);}"
            "</style>"
        )


def _box_metrics(
    axis: Any, tokens: Sequence[Any], fontsize: float
) -> tuple[list[float], float, float]:
    """Return rendered box widths, the box padding and the gap in pixels."""
    pad = _BOX_PAD * fontsize * float(axis.figure.dpi) / 72.0
    gap = 0.5 * fontsize * float(axis.figure.dpi) / 72.0
    widths = [float(token.get_window_extent().width) + 2.0 * pad for token in tokens]
    return widths, pad, gap


def _wrap_tokens(axis: Any, tokens: Sequence[Any], fontsize: float) -> list[list[int]]:
    """Group token indices into lines that fit the axes width, measured in pixels."""
    widths, _, gap = _box_metrics(axis, tokens, fontsize)
    limit = float(axis.get_window_extent().width) * 0.98
    lines: list[list[int]] = [[]]
    used = 0.0
    for index, width in enumerate(widths):
        if lines[-1] and used + width > limit:
            lines.append([])
            used = 0.0
        lines[-1].append(index)
        used += width + gap
    return lines


def _place_tokens(
    axis: Any, tokens: Sequence[Any], lines: list[list[int]], fontsize: float
) -> None:
    """Position tokens left to right per line so each box clears the next."""
    widths, pad, gap = _box_metrics(axis, tokens, fontsize)
    bounds = axis.get_window_extent()
    axis_width, axis_height = float(bounds.width), float(bounds.height)
    line_px = _LINE_INCHES * float(axis.figure.dpi)
    for row, line in enumerate(lines):
        cursor = 0.01 * axis_width
        y_frac = 1.0 - (row + 0.5) * line_px / axis_height
        for index in line:
            tokens[index].set_position(((cursor + pad) / axis_width, y_frac))
            cursor += widths[index] + gap


def _draw_color_legend(axis: Any) -> None:
    axis.text(
        0.02,
        0.08,
        "● increases",
        transform=axis.transAxes,
        fontsize=9,
        color="#FF0D57",
        ha="left",
        va="center",
    )
    axis.text(
        0.16,
        0.08,
        "● decreases",
        transform=axis.transAxes,
        fontsize=9,
        color="#1E88E5",
        ha="left",
        va="center",
    )


def _escape_html(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
