"""Embedding-based value functions.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_embedding.py]]
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ..domain.enums import EmbeddingMode
from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.embedding import EmbeddingProvider
from ..protocols.plugin import PluginOptions
from ._metrics import cosine_similarity, euclidean_similarity

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class EmbeddingValue(ABC):
    """Similarity between static or contextual embedding vectors."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    registered_name: ClassVar[str]
    """Registered value-function identifier."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        embedding_mode: EmbeddingMode = EmbeddingMode.STATIC,
        provider: EmbeddingProvider | None = None,
    ) -> None:
        """Pick the mode; a ``context`` supplies ``explanation.embedding_mode``."""
        if context is not None:
            embedding_mode = context.config.explanation.embedding_mode
        self._embedding_mode = embedding_mode
        self._provider = provider

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return self.registered_name

    @property
    def needs_embeddings(self) -> bool:
        """Always true; scoring reads the records' embedding vectors."""
        return True

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the similarity of the two embedding vectors."""
        return self._similarity(
            _resolve_embedding(base, self._embedding_mode, self._provider),
            _resolve_embedding(candidate, self._embedding_mode, self._provider),
        )

    @abstractmethod
    def _similarity(self, first: tuple[float, ...], second: tuple[float, ...]) -> float:
        """Return the similarity of two vectors."""


class CosineEmbeddingValue(EmbeddingValue):
    """Cosine similarity between static or contextual embeddings (U1/U2)."""

    registered_name = "embedding_cosine"

    def _similarity(self, first: tuple[float, ...], second: tuple[float, ...]) -> float:
        return cosine_similarity(first, second)


class EuclideanEmbeddingValue(EmbeddingValue):
    """Similarity ``1 / (1 + distance)`` between embeddings (U4)."""

    registered_name = "embedding_euclidean"

    def _similarity(self, first: tuple[float, ...], second: tuple[float, ...]) -> float:
        return euclidean_similarity(first, second)


def _resolve_embedding(
    generation: GenerationResult,
    embedding_mode: EmbeddingMode,
    provider: EmbeddingProvider | None,
) -> tuple[float, ...]:
    if isinstance(generation, GenerationRecord):
        if embedding_mode == EmbeddingMode.CONTEXTUAL:
            if generation.contextual_embedding:
                return generation.contextual_embedding
            if generation.embedding:
                return generation.embedding
        elif generation.embedding:
            return generation.embedding
    if provider is not None:
        return provider.embed(generation)
    msg = "embedding vector or EmbeddingProvider is required"
    raise TypeError(msg)
