"""Player partition plugins: whitespace tokens, sentences and whole messages.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_partitions.py]]
"""

import re
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import Field

from ..domain.conversation import ConversationSnapshot, Message
from ..domain.enums import RoleName
from ..domain.players import PlayerSet, PlayerSpan
from ..protocols.plugin import PluginOptions
from .tokens import token_spans

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class RolesOptions(PluginOptions):
    """Options of the ``tokens`` and ``messages`` partitions."""

    roles: list[RoleName] = Field(default=["user", "assistant", "system"])
    """Roles whose text becomes players; other messages are always present."""


MessagesOptions = RolesOptions
"""Options of the ``messages`` partition (same fields as ``tokens``)."""


def _roles(context: "PluginContext | None") -> frozenset[str]:
    roles = RolesOptions().roles
    if context is not None:
        roles = context.option("roles", roles)
    return frozenset(roles)


class TokenPartitioner:
    """Partition a snapshot into whitespace-delimited text tokens."""

    options_model: ClassVar[type[PluginOptions]] = RolesOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read option ``roles`` from ``context``; every role by default."""
        self.roles = _roles(context)
        """Roles whose tokens become players."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "tokens"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive ordered token players and their spans from the snapshot text.

        Tokens of messages whose role is not in :attr:`roles` are no players and
        stay present in every coalition; player ids do not depend on the filter.
        """
        messages = snapshot.messages()
        pairs = tuple(
            (player_id, span)
            for player_id, span in token_spans(snapshot)
            if messages[span.message_index].role.value in self.roles
        )
        if not pairs:
            msg = "snapshot must contain at least one explainable token"
            raise ValueError(msg)
        return PlayerSet(
            player_ids=tuple(player_id for player_id, _ in pairs),
            spans=tuple(span for _, span in pairs),
        )


_PARAGRAPH_RE = re.compile(r"(?:[^\n]|\n(?!\s*\n))+")
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


class SentencePartitioner:
    """Partition a snapshot into sentences, never crossing a blank line."""

    options_model: ClassVar[type[PluginOptions]] = RolesOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read option ``roles`` from ``context``; every role by default."""
        self.roles = _roles(context)
        """Roles whose sentences become players."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "sentences"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive one player per sentence of the selected roles.

        A sentence ends at ``.``, ``!`` or ``?`` before whitespace or at the end of
        its paragraph. Player ids are ``{snapshot_id}:{turn}:{message}:{sentence}``.
        """
        ids: list[str] = []
        spans: list[PlayerSpan] = []
        index = 0
        for turn_index, turn in enumerate(snapshot.turns):
            for local_index, message in enumerate(turn.messages):
                if message.role.value in self.roles:
                    for number, (start, end) in enumerate(
                        _sentence_bounds(message.text)
                    ):
                        ids.append(
                            f"{snapshot.snapshot_id}:{turn_index}:{local_index}:{number}"
                        )
                        spans.append(
                            PlayerSpan(
                                message_index=index,
                                start=start,
                                end=end,
                                text=message.text[start:end],
                                speaker=message.speaker,
                            )
                        )
                index += 1
        if not spans:
            msg = "snapshot must contain at least one explainable sentence"
            raise ValueError(msg)
        return PlayerSet(player_ids=tuple(ids), spans=tuple(spans))


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    bounds = []
    for paragraph in _PARAGRAPH_RE.finditer(text):
        offset = paragraph.start()
        for sentence in _SENTENCE_RE.finditer(paragraph.group()):
            end = offset + sentence.end()
            while end > offset + sentence.start() and text[end - 1].isspace():
                end -= 1
            bounds.append((offset + sentence.start(), end))
    return bounds


LABEL_PREVIEW_CHARS: Final[int] = 30
"""Characters of the message text shown in a message player's label."""


class MessagePartitioner:
    """Partition a snapshot into one player per non-empty message."""

    options_model: ClassVar[type[PluginOptions]] = RolesOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read option ``roles`` from ``context``; every role by default."""
        self.roles = _roles(context)
        """Roles whose messages become players."""

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "messages"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Make each selected message a player whose span is its whole text.

        Player ids are ``{snapshot_id}:{turn}:{message}``; labels read
        ``"<role>[/<speaker>] <n>: <first 30 characters>"`` with ``n`` the
        1-based message position.
        """
        ids: list[str] = []
        spans: list[PlayerSpan] = []
        index = 0
        for turn_index, turn in enumerate(snapshot.turns):
            for local_index, message in enumerate(turn.messages):
                if message.text and message.role.value in self.roles:
                    ids.append(f"{snapshot.snapshot_id}:{turn_index}:{local_index}")
                    spans.append(_message_span(index, message))
                index += 1
        if not spans:
            msg = "snapshot must contain at least one non-empty message to explain"
            raise ValueError(msg)
        return PlayerSet(player_ids=tuple(ids), spans=tuple(spans))


def _message_span(index: int, message: Message) -> PlayerSpan:
    head = message.role.value
    if message.speaker is not None:
        head += f"/{message.speaker}"
    preview = message.text[:LABEL_PREVIEW_CHARS]
    if len(message.text) > LABEL_PREVIEW_CHARS:
        preview += "…"
    return PlayerSpan(
        message_index=index,
        start=0,
        end=len(message.text),
        text=message.text,
        speaker=message.speaker,
        label=f"{head} {index + 1}: {preview}",
    )
