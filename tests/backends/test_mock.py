"""Tests for the deterministic mock backend.

@see [[src/mllm_shap/backends/mock/__init__.py]]
"""

import asyncio

from mllm_shap.backends._chat import snapshot_to_chat_messages
from mllm_shap.backends.mock import MockBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def test_mock_backend_is_deterministic() -> None:
    """Identical inputs produce identical generation records."""
    backend = MockBackend(model_id="stub")
    snapshot = _snapshot("hello world")

    async def run() -> tuple[str, str]:
        first = await backend.generate(
            snapshot, GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1)
        )
        second = await backend.generate(
            snapshot, GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1)
        )
        return first.text, second.text

    first_text, second_text = asyncio.run(run())
    assert first_text == second_text


def test_mock_backend_varies_with_masked_text() -> None:
    """Different masked snapshots produce different generations."""
    backend = MockBackend(model_id="stub")
    full = _snapshot("alpha beta gamma")
    partial = _snapshot("alpha gamma")

    async def run() -> tuple[str, str]:
        full_record = await backend.generate(
            full, GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1)
        )
        partial_record = await backend.generate(
            partial, GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1)
        )
        return full_record.text, partial_record.text

    full_text, partial_text = asyncio.run(run())
    assert full_text != partial_text


def test_mock_backend_exposes_model_id() -> None:
    """Backend reports the configured model identifier."""
    backend = MockBackend(model_id="custom-mock")
    assert backend.model_id == "custom-mock"


def test_mock_backend_populates_embeddings() -> None:
    """Mock generations include static and contextual embedding vectors."""
    backend = MockBackend(model_id="stub")
    snapshot = _snapshot("hello world")

    async def run() -> tuple[tuple[float, ...], tuple[float, ...]]:
        record = await backend.generate(
            snapshot, GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1)
        )
        return record.embedding, record.contextual_embedding

    embedding, contextual = asyncio.run(run())
    assert len(embedding) == 8
    assert len(contextual) == 8
    assert embedding != contextual


def test_chat_messages_prefix_the_speaker() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="Be brief."),
                Message(role=Role.USER, text="Hi", speaker="alice"),
                Message(role=Role.USER, text="", speaker="bob"),
            )
        ),
    ))
    # A message emptied by the delete policy does not reveal its speaker.
    assert snapshot_to_chat_messages(snapshot) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "alice: Hi"},
        {"role": "user", "content": ""},
    ]


def test_mock_backend_hashes_the_rendered_speaker() -> None:
    """Moving a speaker name into the text renders the same prompt and output."""
    backend = MockBackend(model_id="stub")
    generation = GenerationConfig(max_new_tokens=4)
    tagged = Message(role=Role.USER, text="Hi", speaker="alice")
    inline = Message(role=Role.USER, text="alice: Hi")
    texts = [
        asyncio.run(
            backend.generate(
                ConversationSnapshot(turns=(Turn(messages=(m,)),), snapshot_id="s"),
                generation,
            )
        ).text
        for m in (tagged, inline)
    ]
    assert texts[0] == texts[1]
