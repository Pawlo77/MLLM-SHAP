"""Tests for LM Studio chat conversion helpers.

@see [[src/mllm_shap/backends/lmstudio/chat.py]]
"""

from mllm_shap.backends.lmstudio.chat import snapshot_to_chat
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role


class _FakeChat:
    def __init__(self, initial_prompt: str | None = None) -> None:
        self.initial_prompt = initial_prompt
        self.messages: list[tuple[str, str]] = []

    def add_system_prompt(self, text: str) -> None:
        self.messages.append(("system", text))

    def add_user_message(self, text: str) -> None:
        self.messages.append(("user", text))

    def add_assistant_response(self, text: str) -> None:
        self.messages.append(("assistant", text))


def test_snapshot_to_chat_maps_roles_in_order() -> None:
    """Conversation messages are mapped to LM Studio chat helpers."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="be concise"),
                Message(role=Role.USER, text="hello"),
                Message(role=Role.ASSISTANT, text="hi"),
            )
        ),
    ))

    chat = snapshot_to_chat(snapshot, _FakeChat)

    assert chat.initial_prompt == "be concise"
    assert chat.messages == [("user", "hello"), ("assistant", "hi")]
