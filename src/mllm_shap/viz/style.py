"""Seaborn-backed styling for attribution figures.

@domain viz
@see [[knowledge/architecture/viz]]
"""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# SHAP default bar / text accent colors (see shap.plots.colors).
SHAP_RED = "#FF0D57"
SHAP_BLUE = "#1E88E5"
MUTED_GRID = "#E6E9EF"
TEXT_MUTED = "#6B7280"
SURFACE = "#FAFBFC"


def apply_attribution_theme() -> None:
    """Apply a clean notebook theme aligned with SHAP plot styling."""
    sns = _import_seaborn()
    sns.set_theme(
        style="white",
        context="notebook",
        font_scale=1.0,
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED_GRID,
            "axes.labelcolor": TEXT_MUTED,
            "xtick.color": TEXT_MUTED,
            "ytick.color": "#374151",
            "grid.color": MUTED_GRID,
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Segoe UI",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
        },
    )


def new_attribution_figure(
    *,
    width: float,
    height: float,
) -> tuple["Figure", "Axes"]:
    """Create a styled figure and single axes for attribution plots."""
    plt = _import_pyplot()
    apply_attribution_theme()
    fig, axis = plt.subplots(figsize=(width, height))
    return cast("Figure", fig), cast("Axes", axis)


def polish_bar_axes(axis: "Axes") -> None:
    """Remove chart junk and keep a subtle vertical grid."""
    axis.axvline(0.0, color=MUTED_GRID, linewidth=1.2, zorder=0)
    axis.grid(axis="x", color=MUTED_GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(MUTED_GRID)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", colors=TEXT_MUTED)


def polish_text_axes(axis: "Axes") -> None:
    """Minimal axes for inline token layouts."""
    axis.set_axis_off()
    axis.set_facecolor("white")


def _import_seaborn() -> Any:
    try:
        import seaborn as sns
    except ImportError as exc:
        msg = "seaborn is required for the visualization extra"
        raise ImportError(msg) from exc
    return sns


def _import_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = "matplotlib is required for the visualization extra"
        raise ImportError(msg) from exc
    return plt
