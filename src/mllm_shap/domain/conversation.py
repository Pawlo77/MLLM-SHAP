"""Immutable conversation snapshots without backend or IO.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_conversation.py]]
"""

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NotRequired, Self, TypedDict, assert_never

from .enums import Role


@dataclass(frozen=True, slots=True)
class TextPart:
    """Text content of a message."""

    text: str
    """Text payload."""


@dataclass(frozen=True, slots=True)
class AudioPart:
    """Audio content of a message, held in memory or referenced by path."""

    source: Path | bytes
    """Encoded audio bytes, or the path of an audio file."""

    sample_rate: int
    """Sample rate of the audio in hertz."""

    def data(self) -> bytes:
        """Return the encoded audio bytes, reading ``source`` when it is a path."""
        if isinstance(self.source, bytes):
            return self.source
        return self.source.read_bytes()


ContentPart = TextPart | AudioPart
"""One part of a message's content."""


@dataclass(frozen=True, slots=True, init=False)
class Message:
    """One message of a conversation: a role and ordered content parts.

    Build text messages with ``Message(role=..., text=...)`` and multimodal
    messages with ``Message(role=..., content=(TextPart(...), AudioPart(...)))``.
    """

    role: Role
    """Participant role for this message."""

    content: tuple[ContentPart, ...]
    """Ordered content parts of the message."""

    speaker: str | None
    """Name of the participant who wrote the message, when roles are shared."""

    def __init__(
        self,
        role: Role,
        text: str | None = None,
        content: tuple[ContentPart, ...] | None = None,
        speaker: str | None = None,
    ) -> None:
        """Build a message from ``text`` or from ``content`` parts.

        Raises:
            ValueError: Unless exactly one of ``text`` and ``content`` is given,
                or if ``speaker`` is empty or spans several lines.
        """
        if (text is None) == (content is None):
            msg = "message needs exactly one of text or content"
            raise ValueError(msg)
        if speaker is not None and speaker.splitlines() != [speaker]:
            msg = f"speaker must be a non-empty single-line string, got {speaker!r}"
            raise ValueError(msg)
        parts = (TextPart(text),) if text is not None else tuple(content or ())
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", parts)
        object.__setattr__(self, "speaker", speaker)

    @property
    def text(self) -> str:
        """Return the text parts joined with single spaces."""
        return " ".join(
            part.text for part in self.content if isinstance(part, TextPart)
        )

    def with_text(self, text: str) -> "Message":
        """Return a copy whose text parts are replaced by one part holding ``text``.

        Non-text parts keep their order; the new text part takes the position of
        the first text part (or comes first when the message had none).
        """
        parts: list[ContentPart] = []
        placed = False
        for part in self.content:
            if isinstance(part, TextPart):
                if not placed:
                    parts.append(TextPart(text))
                    placed = True
                continue
            parts.append(part)
        if not placed:
            parts.insert(0, TextPart(text))
        return Message(role=self.role, content=tuple(parts), speaker=self.speaker)

    def with_audio(self, audio: AudioPart) -> "Message":
        """Return a copy whose first audio part is replaced by ``audio``.

        Raises:
            ValueError: If the message has no audio part.
        """
        index = next(
            (i for i, part in enumerate(self.content) if isinstance(part, AudioPart)),
            None,
        )
        if index is None:
            msg = "message has no audio part to replace"
            raise ValueError(msg)
        parts = (*self.content[:index], audio, *self.content[index + 1 :])
        return Message(role=self.role, content=parts, speaker=self.speaker)


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversational turn composed of ordered messages."""

    messages: tuple[Message, ...]
    """Ordered messages that make up this turn."""

    def __post_init__(self) -> None:
        """Reject empty turns and snapshots."""
        if not self.messages:
            msg = "turn must contain at least one message"
            raise ValueError(msg)


class PartPayload(TypedDict):
    """JSON form of one content part."""

    type: str
    """``"text"`` or ``"audio"``."""

    text: NotRequired[str]
    """Text of a text part."""

    path: NotRequired[str]
    """Audio file of an audio part, relative to the snapshot's ``base_dir``."""

    data: NotRequired[str]
    """Base64 audio bytes of an audio part, instead of ``path``."""

    sample_rate: NotRequired[int]
    """Sample rate of an audio part in hertz."""


class MessagePayload(TypedDict):
    """JSON form of a message: ``text`` for one text part, else ``content``."""

    role: str
    """Message role, one of ``"user"``, ``"assistant"`` and ``"system"``."""

    text: NotRequired[str]
    """Message text, when the message is a single text part."""

    content: NotRequired[list[PartPayload]]
    """Ordered content parts, for messages with audio or several parts."""

    speaker: NotRequired[str]
    """Speaker name; omitted when the message has none."""


class TurnPayload(TypedDict):
    """JSON form of a turn."""

    messages: list[MessagePayload]
    """Ordered messages of the turn."""


class SnapshotPayload(TypedDict):
    """JSON form of a text conversation snapshot."""

    turns: list[TurnPayload]
    """Ordered turns of the conversation."""

    snapshot_id: NotRequired[str]
    """Snapshot id when it is not the one derived from the content."""


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    """Frozen conversation state used as the explainability input."""

    turns: tuple[Turn, ...]
    """Ordered turns that define the conversation under study."""

    snapshot_id: str
    """Stable identifier used for deduplication and run archives."""

    def __post_init__(self) -> None:
        """Reject empty turns and snapshots."""
        if not self.turns:
            msg = "snapshot must contain at least one turn"
            raise ValueError(msg)
        if not self.snapshot_id:
            msg = "snapshot_id must be non-empty"
            raise ValueError(msg)

    @classmethod
    def from_turns(cls, turns: tuple[Turn, ...]) -> Self:
        """Build a snapshot with a stable content-derived identifier."""
        return cls(turns=turns, snapshot_id=_digest_turns(turns))

    def messages(self) -> tuple[Message, ...]:
        """Return every message in conversation order (turns flattened)."""
        return tuple(message for turn in self.turns for message in turn.messages)

    @classmethod
    def from_json(cls, text: str, base_dir: Path | None = None) -> Self:
        """Parse ``{"turns": [{"messages": [{"role": ..., "text": ...}]}]}``.

        A message may add ``"speaker"``, and gives either ``"text"`` or a
        ``"content"`` list of ``{"type": "text", "text": ...}`` and
        ``{"type": "audio", "path" or "data": ..., "sample_rate": ...}`` parts;
        relative audio paths resolve against ``base_dir`` (default: the working
        directory) and ``"data"`` is base64. An optional top-level
        ``"snapshot_id"`` sets the id; without it the id is derived from the
        content, as in :meth:`from_turns`.

        Raises:
            ValueError: If ``text`` is not valid JSON, a role is unknown, or a
                turn, the snapshot or its id is empty.
            TypeError: If the JSON does not have the documented shape.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            msg = f"snapshot is not valid JSON: {exc}"
            raise ValueError(msg) from exc
        turns = _turns_from_payload(payload, base_dir or Path())
        snapshot_id = payload.get("snapshot_id")
        if snapshot_id is None:
            return cls.from_turns(turns)
        if not isinstance(snapshot_id, str):
            msg = 'snapshot JSON "snapshot_id" must be a string'
            raise TypeError(msg)
        return cls(turns=turns, snapshot_id=snapshot_id)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize the snapshot in the :meth:`from_json` format.

        ``"snapshot_id"`` is written only when it differs from the id derived from
        the content, so :meth:`from_json` restores the same id either way. Audio
        held in memory is written as base64 ``"data"``, audio files as ``"path"``.
        """
        payload: SnapshotPayload = {
            "turns": [
                {"messages": [_message_to_payload(m) for m in turn.messages]}
                for turn in self.turns
            ]
        }
        if self.snapshot_id != _digest_turns(self.turns):
            payload["snapshot_id"] = self.snapshot_id
        return json.dumps(payload, indent=indent, ensure_ascii=False)


def _message_to_payload(message: Message) -> MessagePayload:
    payload: MessagePayload = {"role": message.role.value}
    if len(message.content) == 1 and isinstance(message.content[0], TextPart):
        payload["text"] = message.text
    else:
        payload["content"] = [_part_to_payload(part) for part in message.content]
    if message.speaker is not None:
        payload["speaker"] = message.speaker
    return payload


def _part_to_payload(part: ContentPart) -> PartPayload:
    match part:
        case TextPart(text=text):
            return {"type": "text", "text": text}
        case AudioPart(source=source, sample_rate=rate):
            if isinstance(source, Path):
                return {"type": "audio", "path": str(source), "sample_rate": rate}
            encoded = base64.b64encode(source).decode("ascii")
            return {"type": "audio", "data": encoded, "sample_rate": rate}
        case _ as unreachable:
            assert_never(unreachable)


def _turns_from_payload(payload: object, base_dir: Path) -> tuple[Turn, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("turns"), list):
        msg = 'snapshot JSON needs a "turns" list'
        raise TypeError(msg)
    turns: list[Turn] = []
    for turn_index, turn in enumerate(payload["turns"]):
        if not isinstance(turn, dict) or not isinstance(turn.get("messages"), list):
            msg = f'turn {turn_index} needs a "messages" list'
            raise TypeError(msg)
        messages: list[Message] = []
        messages = [
            _message_from_payload(m, turn_index, base_dir) for m in turn["messages"]
        ]
        turns.append(Turn(messages=tuple(messages)))
    return tuple(turns)


def _message_from_payload(message: object, turn_index: int, base_dir: Path) -> Message:
    fields = message if isinstance(message, dict) else {}
    role, text = fields.get("role"), fields.get("text")
    content, speaker = fields.get("content"), fields.get("speaker")
    if not isinstance(role, str) or (text is None) == (content is None):
        msg = (
            f'turn {turn_index}: each message needs a string "role" and exactly one '
            'of "text" and "content"'
        )
        raise TypeError(msg)
    if speaker is not None and not isinstance(speaker, str):
        msg = f'turn {turn_index}: message "speaker" must be a string'
        raise TypeError(msg)
    try:
        parsed_role = Role(role)
    except ValueError as exc:
        msg = f"turn {turn_index}: unknown role {role!r}"
        raise ValueError(msg) from exc
    if content is None:
        if not isinstance(text, str):
            msg = f'turn {turn_index}: message "text" must be a string'
            raise TypeError(msg)
        return Message(role=parsed_role, text=text, speaker=speaker)
    if not isinstance(content, list):
        msg = f'turn {turn_index}: message "content" must be a list of parts'
        raise TypeError(msg)
    parts = tuple(_part_from_payload(part, turn_index, base_dir) for part in content)
    return Message(role=parsed_role, content=parts, speaker=speaker)


def _part_from_payload(part: object, turn_index: int, base_dir: Path) -> ContentPart:
    fields = part if isinstance(part, dict) else {}
    kind = fields.get("type")
    if kind == "text":
        if not isinstance(fields.get("text"), str):
            msg = f'turn {turn_index}: a text part needs a string "text"'
            raise TypeError(msg)
        return TextPart(fields["text"])
    if kind != "audio":
        msg = f"turn {turn_index}: unknown content part type {kind!r}"
        raise ValueError(msg)
    rate, path, data = fields.get("sample_rate"), fields.get("path"), fields.get("data")
    if not isinstance(rate, int) or isinstance(rate, bool):
        msg = f'turn {turn_index}: an audio part needs an integer "sample_rate"'
        raise TypeError(msg)
    if isinstance(path, str) and data is None:
        return AudioPart(base_dir / path, rate)
    if isinstance(data, str) and path is None:
        try:
            return AudioPart(base64.b64decode(data, validate=True), rate)
        except binascii.Error as exc:
            msg = f'turn {turn_index}: audio "data" is not valid base64'
            raise ValueError(msg) from exc
    msg = f'turn {turn_index}: an audio part needs exactly one of "path" and "data"'
    raise TypeError(msg)


def _digest_turns(turns: tuple[Turn, ...]) -> str:
    payload = [
        [_message_payload(message) for message in turn.messages] for turn in turns
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _message_payload(message: Message) -> dict[str, object]:
    # A single text part keeps the original {"role", "text"} payload so the ids
    # of text-only snapshots do not change; a speaker is hashed only when set.
    speaker = {} if message.speaker is None else {"speaker": message.speaker}
    if len(message.content) == 1 and isinstance(message.content[0], TextPart):
        return {"role": message.role.value, "text": message.content[0].text, **speaker}
    parts: list[dict[str, object]] = []
    for part in message.content:
        match part:
            case TextPart(text=text):
                parts.append({"type": "text", "text": text})
            case AudioPart(sample_rate=sample_rate):
                digest = hashlib.sha256(part.data()).hexdigest()
                parts.append({
                    "type": "audio",
                    "sha256": digest,
                    "sample_rate": sample_rate,
                })
            case _ as unreachable:
                assert_never(unreachable)
    return {"role": message.role.value, "parts": parts, **speaker}
