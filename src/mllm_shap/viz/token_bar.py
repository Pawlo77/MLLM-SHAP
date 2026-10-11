"""Horizontal bar chart renderer for token attributions.

@domain viz
@see [[knowledge/architecture/viz]]
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from ..domain.estimands import Estimand
from ..protocols.plugin import PluginOptions
from .colors import bar_color
from .labels import check_attribution
from .style import new_attribution_figure, polish_bar_axes

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..plugins.context import PluginContext


class TokenBarRenderer:
    """Render attributions as a SHAP-style horizontal bar chart."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def name(self) -> str:
        """Return the registered renderer identifier."""
        return "token_bar"

    def render(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> "Figure":
        """Build a horizontal bar chart sorted by absolute attribution."""
        check_attribution(labels, values)

        ordering = sorted(
            range(len(labels)),
            key=lambda index: abs(values[index]),
            reverse=True,
        )
        ordered_labels = [labels[index] for index in ordering]
        ordered_values = [values[index] for index in ordering]
        height = max(3.0, len(labels) * 0.55)
        fig, axis = new_attribution_figure(width=9.2, height=height)

        y_positions = list(range(len(ordered_labels)))
        bars = axis.barh(
            y_positions,
            ordered_values,
            height=0.68,
            color=[bar_color(value) for value in ordered_values],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        axis.set_yticks(y_positions, ordered_labels)
        axis.set_xlabel("Attribution", color="#6B7280", fontsize=11)
        axis.set_title(
            title or f"{estimand.value.title()} token attributions",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color="#111827",
            pad=14,
        )
        polish_bar_axes(axis)
        _add_sign_legend(axis)
        fig.tight_layout()
        _annotate_bars(axis, bars, ordered_values)
        return fig


def _annotate_bars(axis: Any, bars: Any, values: Sequence[float]) -> None:
    """Write each value beyond its bar end and widen the x range to hold the text.

    Labels are offset in points, so their rendered width is known before the x range
    is final. The range gets exactly the room the widest negative and positive labels
    need, which keeps a negative label off the y tick labels.
    """
    offset_pt = 4.0
    dpi = float(axis.figure.dpi)
    offset_px = offset_pt * dpi / 72.0
    pad_left = 0.0
    pad_right = 0.0
    for bar, value in zip(bars, values, strict=True):
        negative = float(bar.get_width()) < 0
        label = axis.annotate(
            f"{value:+.3f}",
            xy=(float(bar.get_width()), bar.get_y() + bar.get_height() / 2),
            xytext=(-offset_pt if negative else offset_pt, 0.0),
            textcoords="offset points",
            va="center",
            ha="right" if negative else "left",
            fontsize=9,
            color="#4B5563",
        )
        needed = float(label.get_window_extent().width) + offset_px + 4.0
        if negative:
            pad_left = max(pad_left, needed)
        else:
            pad_right = max(pad_right, needed)
    low = min(0.0, *values)
    high = max(0.0, *values)
    span = (high - low) or 1.0
    axis_px = float(axis.get_window_extent().width)
    px_per_unit = max(axis_px - pad_left - pad_right, 1.0) / span
    axis.set_xlim(low - pad_left / px_per_unit, high + pad_right / px_per_unit)


def _add_sign_legend(axis: Any) -> None:
    from matplotlib.patches import Patch

    legend = axis.legend(
        handles=[
            Patch(facecolor="#FF0D57", edgecolor="none", label="Positive"),
            Patch(facecolor="#1E88E5", edgecolor="none", label="Negative"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color("#6B7280")
