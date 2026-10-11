"""Deterministic mock backend for CI-safe E2E tests.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_mock.py]]
"""

import hashlib
import struct
from typing import TYPE_CHECKING, ClassVar

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...protocols.plugin import PluginOptions
from ...value.generation_records import generation_record_from_text
from .._chat import message_chat_text

if TYPE_CHECKING:
    from ...plugins.context import PluginContext


class MockBackend:
    """Hash-based generative backend with no external model dependencies."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        *,
        model_id: str = "mock",
    ) -> None:
        """Use ``context.config.backend.model_id`` when a context is given."""
        self._model_id = (
            context.config.backend.model_id if context is not None else model_id
        )

    @property
    def model_id(self) -> str:
        """Return the backend model identifier."""
        return self._model_id

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        """Return deterministic generation output for ``snapshot``."""
        text = _deterministic_text(snapshot, self._model_id, generation)
        return generation_record_from_text(text)


def _deterministic_text(
    snapshot: ConversationSnapshot,
    model_id: str,
    generation: GenerationConfig,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(snapshot.snapshot_id.encode("utf-8"))
    for turn in snapshot.turns:
        for message in turn.messages:
            hasher.update(message_chat_text(message).encode("utf-8"))
    hasher.update(model_id.encode("utf-8"))
    hasher.update(
        struct.pack(
            "<ifi", generation.max_new_tokens, generation.temperature, generation.top_k
        ),
    )
    digest = hasher.hexdigest()
    token_count = max(1, min(generation.max_new_tokens, 16))
    return " ".join(digest[index : index + 4] for index in range(0, token_count * 4, 4))
