"""Whitespace token spans and span-based snapshot rendering.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_tokens.py]]
"""

import re
from collections.abc import Sequence

from ..domain.conversation import ConversationSnapshot, Message, Turn
from ..domain.players import PlayerSpan

_TOKEN_RE = re.compile(r"\S+")


def token_spans(snapshot: ConversationSnapshot) -> tuple[tuple[str, PlayerSpan], ...]:
    """Return ``(player_id, span)`` for every whitespace token of ``snapshot``.

    Player ids have the form ``{snapshot_id}:{turn}:{message}:{token}``.
    """
    result: list[tuple[str, PlayerSpan]] = []
    message_index = 0
    for turn_index, turn in enumerate(snapshot.turns):
        for local_index, message in enumerate(turn.messages):
            text = message.text
            for token_index, match in enumerate(_TOKEN_RE.finditer(text)):
                player_id = (
                    f"{snapshot.snapshot_id}:{turn_index}:{local_index}:{token_index}"
                )
                span = PlayerSpan(
                    message_index=message_index,
                    start=match.start(),
                    end=match.end(),
                    text=match.group(),
                    speaker=message.speaker,
                )
                result.append((player_id, span))
            message_index += 1
    return tuple(result)


def render_spans(
    snapshot: ConversationSnapshot,
    spans: Sequence[PlayerSpan],
    replacements: Sequence[str | None],
) -> ConversationSnapshot:
    """Rebuild ``snapshot`` with each span replaced by its replacement text.

    ``None`` deletes a span together with the whitespace that separated it from
    its neighbours, so deleting ``b`` from ``"a b c"`` gives ``"a c"``. Text no
    span covers is kept as is.

    Raises:
        ValueError: If ``spans`` and ``replacements`` differ in length, or a span
            does not match the snapshot text.
    """
    if len(spans) != len(replacements):
        msg = "replacement count must match span count"
        raise ValueError(msg)
    by_message: dict[int, list[tuple[PlayerSpan, str | None]]] = {}
    for span, replacement in zip(spans, replacements, strict=True):
        by_message.setdefault(span.message_index, []).append((span, replacement))

    rebuilt_turns: list[Turn] = []
    message_index = 0
    for turn in snapshot.turns:
        rebuilt: list[Message] = []
        for message in turn.messages:
            edits = by_message.get(message_index)
            if edits is None:
                rebuilt.append(message)
            else:
                rebuilt.append(message.with_text(_render_text(message.text, edits)))
            message_index += 1
        rebuilt_turns.append(Turn(messages=tuple(rebuilt)))
    return ConversationSnapshot.from_turns(tuple(rebuilt_turns))


def _render_text(text: str, edits: list[tuple[PlayerSpan, str | None]]) -> str:
    ordered = sorted(edits, key=lambda edit: edit[0].start)
    gaps: list[str] = []
    cursor = 0
    for span, _ in ordered:
        if span.start < cursor or text[span.start : span.end] != span.text:
            msg = "player spans do not match the snapshot text"
            raise ValueError(msg)
        gaps.append(text[cursor : span.start])
        cursor = span.end
    gaps.append(text[cursor:])

    # Output alternates kept_gaps[0], items[0], kept_gaps[1], ..., kept_gaps[-1].
    kept_gaps = [gaps[0]]
    items: list[str] = []
    last = len(ordered) - 1
    for index, (_, replacement) in enumerate(ordered):
        following = gaps[index + 1]
        if replacement is not None:
            items.append(replacement)
            kept_gaps.append(following)
            continue
        kept_gaps.append(
            _merge_gaps(
                kept_gaps.pop(),
                following,
                at_start=not items,
                at_end=index == last,
            )
        )
    pieces = [kept_gaps[0]]
    for item, gap in zip(items, kept_gaps[1:], strict=True):
        pieces.extend((item, gap))
    return "".join(pieces)


def _merge_gaps(before: str, after: str, at_start: bool, at_end: bool) -> str:
    """Join the gaps around a deleted span without doubling whitespace."""
    if before.strip() or after.strip():
        return before + after
    if at_start:
        return before
    if at_end:
        return after
    return before
