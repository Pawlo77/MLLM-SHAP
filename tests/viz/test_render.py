"""Tests for attribution renderer plugins.

@see [[src/nlp_shap/viz/render.py]]
"""

import logging
import warnings
from itertools import combinations

import pytest

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.estimands import Estimand
from nlp_shap.masking.partitions import TokenPartitioner
from nlp_shap.plugins import PluginGroup, PluginRegistry
from nlp_shap.viz import render_attribution, render_attribution_html
from nlp_shap.viz.colors import diverging_rgba
from nlp_shap.viz.token_bar import TokenBarRenderer
from nlp_shap.viz.token_text import TokenTextRenderer

pytest.importorskip("matplotlib")

_LABELS = ("Refund", "my", "receipt,", "extraordinarily-long-token", "thanks")
_VALUES = (0.632, -0.037, -0.5, 0.3, 0.05)


def _overlap(first: object, second: object) -> bool:
    """Return whether two matplotlib ``Bbox`` rectangles share interior area."""
    return bool(
        first.x0 < second.x1  # type: ignore[attr-defined]
        and second.x0 < first.x1  # type: ignore[attr-defined]
        and first.y0 < second.y1  # type: ignore[attr-defined]
        and second.y0 < first.y1  # type: ignore[attr-defined]
    )


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def _three_token_output() -> tuple[object, ConversationSnapshot, object]:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "generation": {
            "max_new_tokens": 4,
            "temperature": 0.0,
        },
        "explanation": {
            "estimator": "exact",
            "estimand": "shapley",
            "value_fn": "tfidf_cosine",
            "normalizer": "identity",
            "players": "tokens",
            "absence_policy": "pad",
            "budget": {"fraction": 1.0},
            "seed": 0,
            "archive": {"path": ""},
        },
    })
    snapshot = _snapshot("a b c")
    output = ExplainRunner(config).explain_sync(snapshot)
    player_set = TokenPartitioner().partition(snapshot)
    return output, snapshot, player_set


def test_renderer_entry_points_resolve() -> None:
    """Packaging entry points expose token renderers."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.RENDERERS)
    assert registry.names(PluginGroup.RENDERERS) == ("token_bar", "token_text")


def test_token_text_renderer_assigns_distinct_signed_colors() -> None:
    """Positive and negative tokens map to different color channels."""
    positive = diverging_rgba(1.0, 1.0)
    negative = diverging_rgba(-1.0, 1.0)
    assert positive != negative


def test_render_attribution_text_figure_is_non_empty() -> None:
    """Toy exact run renders a token-text figure."""
    output, snapshot, player_set = _three_token_output()
    figure = render_attribution(output, snapshot, player_set, renderer="token_text")
    assert len(figure.axes) == 1
    assert figure.axes[0].texts


def test_token_bar_renderer_sorts_by_absolute_value() -> None:
    """Horizontal bars follow descending absolute attribution order."""
    renderer = TokenBarRenderer()
    figure = renderer.render(
        ("a", "b", "c"),
        (0.1, -0.9, 0.4),
        estimand=Estimand.SHAPLEY,
    )
    bar_container = figure.axes[0].containers[0]
    heights = [bar.get_width() for bar in bar_container]
    assert heights == sorted(heights, key=abs, reverse=True)


def test_render_attribution_html_includes_tokens() -> None:
    """HTML renderer emits colored spans for each token."""
    output, snapshot, player_set = _three_token_output()
    html = render_attribution_html(output, snapshot, player_set)
    assert "a" in html
    assert "background-color:rgba" in html


def test_token_text_renderer_rejects_length_mismatch() -> None:
    """Mismatched labels and values raise ValueError."""
    renderer = TokenTextRenderer()
    with pytest.raises(ValueError, match="same length"):
        renderer.render(("a",), (0.1, 0.2), estimand=Estimand.SHAPLEY)


def test_render_attribution_reuses_runner_registry() -> None:
    """Custom renderers registered on the runner's registry are found."""
    rendered: list[tuple[str, ...]] = []

    class RecordingRenderer:
        def __init__(self, context: object | None = None) -> None:
            _ = context

        @property
        def name(self) -> str:
            return "recording"

        def render(self, labels, values, *, estimand, title=None):
            rendered.append(tuple(labels))
            return "figure"

    output, snapshot, player_set = _three_token_output()
    runner = ExplainRunner(output.manifest.config)
    runner.registry.register(PluginGroup.RENDERERS, "recording", RecordingRenderer)

    figure = render_attribution(
        output, snapshot, player_set, renderer="recording", registry=runner.registry
    )

    assert figure == "figure"
    assert rendered == [("a", "b", "c")]


@pytest.mark.parametrize("renderer", [TokenTextRenderer, TokenBarRenderer])
def test_renderers_emit_no_font_warning(
    renderer: type, caplog: pytest.LogCaptureFixture
) -> None:
    """Rendering with the default fonts logs and warns nothing about font weights."""
    import matplotlib.font_manager as font_manager

    font_manager.fontManager._findfont_cached.cache_clear()  # type: ignore[attr-defined]
    with (
        warnings.catch_warnings(record=True) as caught,
        caplog.at_level(logging.WARNING, logger="matplotlib.font_manager"),
    ):
        warnings.simplefilter("always")
        figure = renderer().render(_LABELS, _VALUES, estimand=Estimand.SHAPLEY)
        figure.canvas.draw()
    assert not caught
    assert not [
        record for record in caplog.records if record.name == "matplotlib.font_manager"
    ]


def test_token_text_boxes_do_not_overlap() -> None:
    """Highlight boxes clear each other, including after a long token wraps lines."""
    labels = _LABELS * 3
    values = _VALUES * 3
    figure = TokenTextRenderer().render(labels, values, estimand=Estimand.SHAPLEY)
    figure.canvas.draw()
    boxes = [
        text.get_bbox_patch().get_window_extent()
        for text in figure.axes[0].texts
        if text.get_text() in labels
    ]
    assert len(boxes) == len(labels)
    assert not [pair for pair in combinations(boxes, 2) if _overlap(*pair)]
    axes_box = figure.axes[0].get_window_extent()
    assert all(box.x1 <= axes_box.x1 for box in boxes)


def test_token_bar_labels_and_legend_do_not_overlap() -> None:
    """Value labels clear the y tick labels, each other and the legend."""
    figure = TokenBarRenderer().render(_LABELS, _VALUES, estimand=Estimand.SHAPLEY)
    figure.canvas.draw()
    axis = figure.axes[0]
    value_labels = [
        text.get_window_extent() for text in axis.texts if text.get_text()[:1] in "+-"
    ]
    tick_labels = [
        tick.get_window_extent() for tick in axis.get_yticklabels() if tick.get_text()
    ]
    legend = axis.get_legend().get_window_extent()
    assert len(value_labels) == len(_VALUES)
    assert len(tick_labels) == len(_LABELS)
    parts = [*value_labels, *tick_labels, legend]
    assert not [pair for pair in combinations(parts, 2) if _overlap(*pair)]
    axes_box = axis.get_window_extent()
    assert all(
        axes_box.x0 <= label.x0 and label.x1 <= axes_box.x1 for label in value_labels
    )
