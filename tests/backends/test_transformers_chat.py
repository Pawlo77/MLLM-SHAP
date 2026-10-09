"""Tests for transformers chat helpers.

@see [[src/nlp_shap/backends/transformers/chat.py]]
"""

from nlp_shap.backends._chat import snapshot_has_audio, snapshot_to_chat_messages
from nlp_shap.backends.transformers.chat import render_prompt
from nlp_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    Turn,
)
from nlp_shap.domain.enums import Role


class _TemplateTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del tokenize, add_generation_prompt
        return "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


def test_snapshot_to_chat_messages_maps_roles() -> None:
    """Conversation roles are mapped to chat-template roles."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="sys"),
                Message(role=Role.USER, text="user"),
                Message(role=Role.ASSISTANT, text="assistant"),
            )
        ),
    ))

    messages = snapshot_to_chat_messages(snapshot)

    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "assistant"},
    ]


def test_render_prompt_uses_chat_template_when_available() -> None:
    """Tokenizers with chat templates render through apply_chat_template."""
    rendered = render_prompt(
        _TemplateTokenizer(),
        [{"role": "user", "content": "hello"}],
    )
    assert rendered == "user:hello"


def test_snapshot_has_audio_detects_audio_part() -> None:
    """Audio snapshots are rejected by the text-only backend."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(Message(role=Role.USER, content=(AudioPart(b"\x00", 16_000),)),)
        ),
    ))
    assert snapshot_has_audio(snapshot) is True


def test_snapshot_has_audio_returns_false_for_text_only() -> None:
    """Plain text snapshots are accepted by the text backend."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="plain text"),)),
    ))
    assert snapshot_has_audio(snapshot) is False
