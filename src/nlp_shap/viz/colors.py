"""SHAP-inspired diverging colors for signed attributions.

@domain viz
@see [[knowledge/architecture/viz]]
"""

from typing import cast

WHITE = (1.0, 1.0, 1.0)
SHAP_RED_RGB = (1.0, 13 / 255, 87 / 255)
SHAP_BLUE_RGB = (30 / 255, 136 / 255, 229 / 255)


def diverging_rgba(value: float, vmax: float) -> tuple[float, float, float, float]:
    """Map a signed attribution to an RGBA tuple (SHAP red / blue scale)."""
    return token_background_rgba(value, vmax)


def token_background_rgba(
    value: float, vmax: float
) -> tuple[float, float, float, float]:
    """Return a soft highlight color for one token (white center, saturated tails)."""
    if vmax <= 0.0:
        return (0.97, 0.97, 0.98, 1.0)
    scaled = 0.5 + 0.5 * max(-1.0, min(1.0, value / vmax))
    if scaled <= 0.5:
        intensity = 1.0 - (scaled / 0.5)
        red, green, blue = _blend(SHAP_BLUE_RGB, WHITE, intensity * 0.92)
        alpha = 0.28 + 0.5 * intensity
    else:
        intensity = (scaled - 0.5) / 0.5
        red, green, blue = _blend(SHAP_RED_RGB, WHITE, intensity * 0.92)
        alpha = 0.28 + 0.5 * intensity
    return (red, green, blue, alpha)


def bar_color(value: float) -> str:
    """Return the SHAP primary color for a signed bar attribution."""
    return "#FF0D57" if value >= 0 else "#1E88E5"


def rgba_css(red: float, green: float, blue: float, alpha: float) -> str:
    """Format an rgba() string for HTML/CSS."""
    return f"rgba({int(red * 255)},{int(green * 255)},{int(blue * 255)},{alpha:.2f})"


def token_background_css(value: float, vmax: float) -> str:
    """Return a CSS background color for one token span."""
    red, green, blue, alpha = token_background_rgba(value, vmax)
    return rgba_css(red, green, blue, alpha)


def _blend(
    foreground: tuple[float, float, float],
    background: tuple[float, float, float],
    weight: float,
) -> tuple[float, float, float]:
    clamped = max(0.0, min(1.0, weight))
    return cast(
        tuple[float, float, float],
        tuple(
            background[channel] * (1.0 - clamped) + foreground[channel] * clamped
            for channel in range(3)
        ),
    )
