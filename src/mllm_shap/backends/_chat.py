"""Conversation helpers shared by chat-style backends.

@domain backends
@see [[knowledge/architecture/backends]]
"""

from ..domain.conversation import AudioPart, ConversationSnapshot, Message
from ..domain.enums import Role


def snapshot_has_audio(snapshot: ConversationSnapshot) -> bool:
    """Return whether any message of ``snapshot`` has an audio part."""
    return any(
        isinstance(part, AudioPart)
        for message in snapshot.messages()
        for part in message.content
    )


def message_chat_text(message: Message) -> str:
    """Return the text the model sees: ``"<speaker>: <text>"`` when a speaker is set.

    Chat templates have no speaker slot, so the name goes into the content. An
    empty message (deleted by the ``delete`` policy) stays empty.
    """
    if message.speaker is None or not message.text:
        return message.text
    return f"{message.speaker}: {message.text}"


def chat_role(role: Role) -> str:
    """Return the chat-template role name (``Role`` values are those names)."""
    return role.value


def snapshot_to_chat_messages(snapshot: ConversationSnapshot) -> list[dict[str, str]]:
    """Convert a conversation snapshot into OpenAI-style chat messages."""
    return [
        {"role": chat_role(message.role), "content": message_chat_text(message)}
        for message in snapshot.messages()
    ]
