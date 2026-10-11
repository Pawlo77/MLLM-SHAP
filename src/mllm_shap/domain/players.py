"""Player indexing for cooperative-game explainability.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_players.py]]
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerSpan:
    """Where one player sits in the conversation it was partitioned from."""

    message_index: int
    """Index of the message in conversation order (turns flattened)."""

    start: int
    """Start character offset of the player inside ``Message.text``."""

    end: int
    """End character offset (exclusive) of the player inside ``Message.text``."""

    text: str
    """Text the player covers, ``Message.text[start:end]``."""

    speaker: str | None = None
    """Speaker of the message the player sits in."""

    label: str | None = None
    """Display label set by the partitioner; :attr:`display_label` falls back
    to the text."""

    samples: tuple[int, int] | None = None
    """Sample range ``[start, end)`` of an audio player in the message's first
    :class:`~mllm_shap.AudioPart`; ``None`` for text-only players."""

    @property
    def display_label(self) -> str:
        """Return :attr:`label`, else the text prefixed with ``"<speaker>: "``."""
        if self.label is not None:
            return self.label
        return self.text if self.speaker is None else f"{self.speaker}: {self.text}"


@dataclass(frozen=True, slots=True)
class PlayerSet:
    """Ordered explainability players aligned with coalition masks."""

    player_ids: tuple[str, ...]
    """Stable player identifiers in coalition-mask order."""

    spans: tuple[PlayerSpan, ...] = ()
    """Player locations in mask order, filled by the partitioner.

    Absence policies and label helpers need them; estimators do not.
    """

    def __post_init__(self) -> None:
        """Reject empty, duplicated or span-misaligned player sets."""
        if not self.player_ids:
            msg = "player set must contain at least one player"
            raise ValueError(msg)
        if len(set(self.player_ids)) != len(self.player_ids):
            msg = "player ids must be unique"
            raise ValueError(msg)
        if self.spans and len(self.spans) != len(self.player_ids):
            msg = "player set needs one span per player"
            raise ValueError(msg)

    @property
    def num_players(self) -> int:
        """Return the number of players in the set."""
        return len(self.player_ids)
