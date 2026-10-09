"""Player partition plugins: whitespace tokens and whole messages.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_partitions.py]]
"""

from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import Field

from ..domain.conversation import ConversationSnapshot, Message
from ..domain.players import PlayerSet, PlayerSpan
from ..protocols.plugin import PluginOptions
from .tokens import token_spans

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class TokenPartitioner:
    """Partition a snapshot into whitespace-delimited text tokens."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "tokens"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Derive ordered token players and their spans from the snapshot text."""
        pairs = token_spans(snapshot)
        if not pairs:
            msg = "snapshot must contain at least one explainable token"
            raise ValueError(msg)
        return PlayerSet(
            player_ids=tuple(player_id for player_id, _ in pairs),
            spans=tuple(span for _, span in pairs),
        )


LABEL_PREVIEW_CHARS: Final[int] = 30
"""Characters of the message text shown in a message player's label."""

RoleName = Literal["user", "assistant", "system"]
"""Message role accepted by :class:`MessagesOptions`."""


class MessagesOptions(PluginOptions):
    """Options of the ``messages`` partition."""

    roles: list[RoleName] = Field(default=["user", "assistant", "system"])
    """Roles whose messages become players; other messages are always present."""


class MessagePartitioner:
    """Partition a snapshot into one player per non-empty message."""

    options_model: ClassVar[type[PluginOptions]] = MessagesOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read option ``roles`` from ``context``; every role by default."""
        roles = MessagesOptions().roles
        if context is not None:
            roles = context.option("roles", roles)
        self.roles = frozenset(roles)
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
