"""Render audio players: cut or silence absent samples, hide alignment transcripts.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/alignment/test_sgpa_players.py]]
"""

from collections.abc import Sequence

import numpy as np

from ..alignment.io import decode_audio, encode_wav
from ..domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from ..domain.players import PlayerSpan


def render_audio(
    snapshot: ConversationSnapshot,
    spans: Sequence[PlayerSpan],
    present: Sequence[bool],
    cut: bool,
) -> ConversationSnapshot:
    """Remove (``cut``) or silence the samples of absent players with ``samples``.

    Silencing keeps the audio's length and timing; cutting joins what is left.
    Spans without ``samples`` and messages without absent audio are untouched.
    """
    absent: dict[int, list[tuple[int, int]]] = {}
    for span, keep in zip(spans, present, strict=True):
        if span.samples is not None and not keep:
            absent.setdefault(span.message_index, []).append(span.samples)
    if not absent:
        return snapshot
    turns: list[Turn] = []
    index = 0
    for turn in snapshot.turns:
        messages: list[Message] = []
        for message in turn.messages:
            ranges = absent.get(index)
            messages.append(message if ranges is None else _edit(message, ranges, cut))
            index += 1
        turns.append(Turn(messages=tuple(messages)))
    return ConversationSnapshot.from_turns(tuple(turns))


def hide_text(
    snapshot: ConversationSnapshot, message_indices: Sequence[int]
) -> ConversationSnapshot:
    """Drop the text parts of the messages at ``message_indices`` (turns flattened)."""
    if not message_indices:
        return snapshot
    hidden = set(message_indices)
    turns: list[Turn] = []
    index = 0
    for turn in snapshot.turns:
        messages: list[Message] = []
        for message in turn.messages:
            if index in hidden:
                parts = tuple(p for p in message.content if not isinstance(p, TextPart))
                message = Message(
                    role=message.role, content=parts, speaker=message.speaker
                )
            messages.append(message)
            index += 1
        turns.append(Turn(messages=tuple(messages)))
    return ConversationSnapshot.from_turns(tuple(turns))


def _edit(message: Message, ranges: list[tuple[int, int]], cut: bool) -> Message:
    part = next(p for p in message.content if isinstance(p, AudioPart))
    waveform, rate = decode_audio(part.data())
    keep = np.ones(len(waveform), dtype=bool)
    for start, end in ranges:
        keep[start:end] = False
    edited = waveform[keep] if cut else np.where(keep, waveform, 0.0)
    return message.with_audio(AudioPart(encode_wav(edited, rate), rate))
