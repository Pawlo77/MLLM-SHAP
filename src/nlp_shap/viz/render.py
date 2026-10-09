"""Public helpers for attribution visualization.

@domain viz
@see [[knowledge/architecture/viz]]
@tested-by [[tests/viz/test_render.py]]
"""

from typing import TYPE_CHECKING, Any, cast

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..pipeline.result import ExplainRunOutput
from ..plugins.builtin import default_registry
from ..plugins.context import PluginContext
from ..plugins.groups import PluginGroup
from ..plugins.registry import PluginRegistry
from ..protocols.renderer import AttributionRenderer
from .labels import token_labels

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def render_attribution(
    output: ExplainRunOutput,
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    *,
    renderer: str = "token_text",
    title: str | None = None,
    registry: PluginRegistry | None = None,
) -> "Figure":
    """Render ``output`` attributions with a registered renderer plugin.

    Pass ``registry=runner.registry`` to reuse the registry of the
    :class:`~nlp_shap.pipeline.runner.ExplainRunner` that produced ``output``
    (needed for custom renderers); otherwise the default registry is built.
    """
    plugin_registry = registry or default_registry()
    config = output.manifest.config
    if config is None:
        msg = "render_attribution needs an output whose manifest records its config"
        raise ValueError(msg)
    attribution_renderer = cast(
        AttributionRenderer,
        plugin_registry.resolve(
            PluginGroup.RENDERERS,
            renderer,
            PluginContext(config=config, registry=plugin_registry),
        ),
    )
    labels = token_labels(snapshot, player_set)
    values = output.result.values
    return cast(
        "Figure",
        attribution_renderer.render(
            labels,
            values,
            estimand=output.result.estimand,
            title=title,
        ),
    )


def render_attribution_html(
    output: ExplainRunOutput,
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    *,
    title: str | None = None,
) -> str:
    """Return an HTML fragment for inline token coloring in Jupyter."""
    from .token_text import TokenTextRenderer

    renderer = TokenTextRenderer()
    labels = token_labels(snapshot, player_set)
    return renderer.to_html(
        labels,
        output.result.values,
        estimand=output.result.estimand,
        title=title,
    )


def display_attribution_html(
    output: ExplainRunOutput,
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    *,
    title: str | None = None,
) -> Any:
    """Display colored token HTML when IPython is available."""
    html = render_attribution_html(output, snapshot, player_set, title=title)
    display_cls = _import_ipython_display()
    return display_cls(html)


def _import_ipython_display() -> Any:
    try:
        from IPython.display import HTML
    except ImportError as exc:
        msg = "IPython is required to display attribution HTML"
        raise ImportError(msg) from exc
    return HTML
