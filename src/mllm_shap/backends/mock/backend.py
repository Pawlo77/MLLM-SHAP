"""Deterministic mock backend for CI-safe E2E tests.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_mock.py]]
"""

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import ForcedScore, GenerationRecord
from ...protocols.plugin import PluginOptions
from ...value.generation_records import generation_record_from_text, stable_token_id
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

    @property
    def forces_continuations(self) -> bool:
        """Always true; scores are deterministic functions of the prompt."""
        return True

    async def score_continuations(
        self,
        snapshot: ConversationSnapshot,
        continuations: Sequence[str],
        top_k: int = 0,
    ) -> list[ForcedScore]:
        """Score continuations under a hash-derived next-token distribution.

        Each continuation word is one of ``MOCK_VOCAB`` tokens; the distribution
        at every position depends on the prompt and the words before it, so
        different coalition prompts give different scores.
        """
        prompt = _prompt_digest(snapshot, self._model_id)
        return [_force(prompt, text, top_k) for text in continuations]


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


MOCK_VOCAB = 8
"""Size of the mock backend's next-token vocabulary."""


def _prompt_digest(snapshot: ConversationSnapshot, model_id: str) -> bytes:
    hasher = hashlib.sha256(model_id.encode("utf-8"))
    for turn in snapshot.turns:
        for message in turn.messages:
            hasher.update(message_chat_text(message).encode("utf-8"))
    return hasher.digest()


def _force(prompt: bytes, continuation: str, top_k: int) -> ForcedScore:
    logprobs: list[float] = []
    ids: list[tuple[int, ...]] = []
    tops: list[tuple[float, ...]] = []
    seen = prompt
    for word in continuation.split():
        digest = hashlib.sha256(seen).digest()
        logits = [digest[i] / 64.0 for i in range(MOCK_VOCAB)]
        norm = math.log(sum(math.exp(x) for x in logits))
        dist = [x - norm for x in logits]
        logprobs.append(dist[stable_token_id(word) % MOCK_VOCAB])
        if top_k > 0:
            order = sorted(range(MOCK_VOCAB), key=lambda i: (-dist[i], i))[:top_k]
            ids.append(tuple(order))
            tops.append(tuple(dist[i] for i in order))
        seen = digest + word.encode("utf-8")
    return ForcedScore(tuple(logprobs), tuple(ids), tuple(tops))
