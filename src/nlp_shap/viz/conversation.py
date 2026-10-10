"""Conversation view: each message under its role, players coloured in place.

@domain viz
@see [[knowledge/architecture/viz]]
@tested-by [[tests/viz/test_conversation_view.py]]
"""

from typing import TYPE_CHECKING, Any

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..pipeline.result import ExplainRunOutput
from .colors import token_background_css, token_background_rgba
from .labels import check_attribution, token_labels
from .style import TEXT_MUTED, new_attribution_figure, polish_text_axes
from .token_text import (
    _BOX_PAD,
    _HEADER_INCHES,
    _LINE_INCHES,
    _draw_color_legend,
    _escape_html,
    _place_tokens,
    _wrap_tokens,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_CONTEXT = "#9CA3AF"  # text that is no player: filtered roles, punctuation gaps
Piece = tuple[str, float | None]


def plot_conversation(
    output: ExplainRunOutput,
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    *,
    title: str | None = None,
) -> "Figure":
    """Draw ``snapshot`` message by message with each player coloured by value.

    Every message gets a ``"<role>[/<speaker>] <n>"`` header. Token or message
    players are shaded on the attribution scale where they stand; text that is
    no player (roles the partition left out, for example) stays grey, so the
    context of a filtered explanation remains readable.

    Raises:
        ValueError: If ``player_set`` does not match ``snapshot`` or the values.
    """
    values = _checked_values(output, snapshot, player_set)
    vmax = max(abs(value) for value in values)
    figure, axis = new_attribution_figure(width=10.0, height=3.0)
    polish_text_axes(axis)
    axis.set_title(
        title or f"{output.result.estimand.value.title()} attributions by message",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color="#111827",
        pad=12,
    )
    fontsize = 11.0
    texts: list[Any] = []
    groups: list[tuple[int, int]] = []  # (header index, first word index) per message
    for head, pieces in _messages(snapshot, player_set, values):
        texts.append(_text(axis, head, 9.0, TEXT_MUTED, weight="bold"))
        first = len(texts)
        for words, value in ((piece.split(), value) for piece, value in pieces):
            fill = None if value is None else token_background_rgba(value, vmax)
            texts.extend(_text(axis, word, fontsize, fill=fill) for word in words)
        groups.append((first - 1, first))
    figure.tight_layout()
    lines = _lines(axis, texts, groups, fontsize)
    figure.set_size_inches(10.0, _HEADER_INCHES + _LINE_INCHES * len(lines))
    figure.tight_layout()
    _place_tokens(axis, texts, lines, fontsize)
    _draw_color_legend(axis)
    return figure


def render_conversation_html(
    output: ExplainRunOutput,
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    *,
    title: str | None = None,
) -> str:
    """Return :func:`plot_conversation` as an HTML fragment for Jupyter or reports.

    Hovering a player shows its value.
    """
    values = _checked_values(output, snapshot, player_set)
    vmax = max(abs(value) for value in values)
    estimand = output.result.estimand.value
    header = title or f"{estimand.title()} attributions by message"
    blocks = []
    for head, pieces in _messages(snapshot, player_set, values):
        body = " ".join(
            _piece_html(piece, value, vmax, estimand) for piece, value in pieces
        )
        blocks.append(
            '<div style="margin:0 0 10px 0;">'
            f'<div style="font-size:12px;font-weight:600;color:{TEXT_MUTED};">'
            f"{_escape_html(head)}</div><div>{body}</div></div>"
        )
    return (
        '<div class="nlp-shap-conversation" style="font-family:system-ui,-apple-system,'
        "Segoe UI,Roboto,Helvetica Neue,Arial,sans-serif;line-height:1.8;"
        'color:#111827;font-size:15px;">'
        f'<div style="font-size:14px;font-weight:600;margin-bottom:8px;">'
        f"{_escape_html(header)}</div>{''.join(blocks)}</div>"
    )


def _checked_values(
    output: ExplainRunOutput, snapshot: ConversationSnapshot, player_set: PlayerSet
) -> tuple[float, ...]:
    values = output.result.values
    check_attribution(token_labels(snapshot, player_set), values)
    return values


def _messages(
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    values: tuple[float, ...],
) -> list[tuple[str, list[Piece]]]:
    """Each message's header and its text cut into players and context."""
    by_message: dict[int, list[tuple[int, int, float]]] = {}
    for span, value in zip(player_set.spans, values, strict=True):
        by_message.setdefault(span.message_index, []).append((
            span.start,
            span.end,
            value,
        ))
    result: list[tuple[str, list[Piece]]] = []
    for index, message in enumerate(snapshot.messages()):
        head = message.role.value
        if message.speaker is not None:
            head += f"/{message.speaker}"
        pieces: list[Piece] = []
        cursor = 0
        for start, end, value in sorted(by_message.get(index, [])):
            pieces.append((message.text[cursor:start], None))
            pieces.append((message.text[start:end], value))
            cursor = end
        pieces.append((message.text[cursor:], None))
        result.append((f"{head} {index + 1}", [p for p in pieces if p[0].strip()]))
    return result


def _text(
    axis: Any,
    text: str,
    fontsize: float,
    color: str = "#111827",
    fill: tuple[float, float, float, float] | None = None,
    weight: str = "normal",
) -> Any:
    bbox = None
    if fill is not None:
        style = f"round,pad={_BOX_PAD},rounding_size=0.08"
        bbox = {"boxstyle": style, "facecolor": fill, "edgecolor": "none"}
    return axis.text(
        0.0,
        0.0,
        text,
        transform=axis.transAxes,
        fontsize=fontsize,
        fontweight=weight,
        va="center",
        ha="left",
        color=color if fill is not None or weight == "bold" else _CONTEXT,
        bbox=bbox,
    )


def _lines(
    axis: Any, texts: list[Any], groups: list[tuple[int, int]], fontsize: float
) -> list[list[int]]:
    """One line per header, then the message's words wrapped to the axes width."""
    lines: list[list[int]] = []
    ends = [first for _, first in groups[1:]] + [len(texts) + 1]
    for (header, first), end in zip(groups, ends, strict=True):
        lines.append([header])
        words = texts[first : end - 1]
        lines.extend(
            [first + i for i in line]
            for line in _wrap_tokens(axis, words, fontsize)
            if line
        )
    return lines


def _piece_html(piece: str, value: float | None, vmax: float, estimand: str) -> str:
    text = _escape_html(piece.strip())
    if value is None:
        return f'<span style="color:{_CONTEXT};">{text}</span>'
    return (
        f'<span class="nlp-shap-player" title="{estimand}: {value:+.3f}" '
        f'style="background-color:{token_background_css(value, vmax)};'
        f'padding:2px 5px;border-radius:4px;">{text}</span>'
    )
