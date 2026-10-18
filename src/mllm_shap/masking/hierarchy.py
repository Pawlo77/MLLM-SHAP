"""Two-level players: sentence groups (paragraphs or messages) and their members.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_hierarchy.py]]
"""

import re
from collections.abc import Sequence
from typing import Literal

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet, PlayerSpan

_BLANK_LINE_RE = re.compile(r"\n\s*\n")
PREVIEW_CHARS = 30
"""Characters of the group text shown in its label."""


def group_players(
    snapshot: ConversationSnapshot,
    players: PlayerSet,
    group_by: Literal["paragraph", "message"],
) -> list[list[int]]:
    """Return the player indices of each group, in text order.

    A group is a paragraph (text between blank lines) of one message, or a whole
    message.
    """
    messages = snapshot.messages()
    groups: dict[tuple[int, int], list[int]] = {}
    for index, span in enumerate(players.spans):
        paragraph = 0
        if group_by == "paragraph":
            text = messages[span.message_index].text
            breaks = _BLANK_LINE_RE.finditer(text)
            paragraph = sum(m.end() <= span.start for m in breaks)
        groups.setdefault((span.message_index, paragraph), []).append(index)
    return [groups[key] for key in sorted(groups)]


def coarse_player_set(
    snapshot: ConversationSnapshot, players: PlayerSet, groups: Sequence[Sequence[int]]
) -> PlayerSet:
    """Return one player per group, spanning its first to its last member."""
    messages = snapshot.messages()
    ids: list[str] = []
    spans: list[PlayerSpan] = []
    for number, members in enumerate(groups):
        first, last = players.spans[members[0]], players.spans[members[-1]]
        text = messages[first.message_index].text[first.start : last.end]
        preview = text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")
        ids.append(f"{snapshot.snapshot_id}:group{number}")
        spans.append(
            PlayerSpan(
                message_index=first.message_index,
                start=first.start,
                end=last.end,
                text=text,
                speaker=first.speaker,
                label=f"group {number + 1}: {preview}",
            )
        )
    return PlayerSet(player_ids=tuple(ids), spans=tuple(spans))


def refined_player_set(
    players: PlayerSet, groups: Sequence[Sequence[int]], kept: Sequence[int]
) -> PlayerSet:
    """Return the members of the kept groups; other text stays present."""
    chosen = [index for group in sorted(kept) for index in groups[group]]
    return PlayerSet(
        player_ids=tuple(players.player_ids[i] for i in chosen),
        spans=tuple(players.spans[i] for i in chosen),
    )
