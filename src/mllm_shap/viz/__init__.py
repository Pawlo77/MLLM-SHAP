"""Token attribution visualization plugins.

@domain viz
@see [[knowledge/architecture/viz]]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .audio import plot_audio_segments
    from .render import (
        display_attribution_html,
        render_attribution,
        render_attribution_html,
    )

__all__ = [
    "display_attribution_html",
    "plot_audio_segments",
    "render_attribution",
    "render_attribution_html",
]


def __getattr__(name: str) -> object:
    # ``render`` imports the pipeline; loading it lazily lets the plugin registry
    # import the renderer classes without an import cycle.
    if name == "plot_audio_segments":
        from . import audio

        return audio.plot_audio_segments
    if name in __all__:
        from . import render

        return getattr(render, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
