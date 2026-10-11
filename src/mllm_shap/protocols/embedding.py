"""Embedding provider protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from typing import Protocol

from .backend import GenerationResult


class EmbeddingProvider(Protocol):
    """Expose embeddings for value functions that operate in embedding space."""

    @property
    def name(self) -> str:
        """Return the registered embedding provider identifier."""

    def embed(self, generation: GenerationResult) -> tuple[float, ...]:
        """Return an embedding vector for the generation."""
