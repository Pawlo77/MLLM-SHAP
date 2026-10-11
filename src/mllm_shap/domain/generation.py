"""Generation payloads scored by value functions.

@domain domain
@see [[knowledge/architecture/domain]]
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRecord:
    """Concrete generation output consumed by built-in value functions."""

    text: str
    """Generated text payload."""

    text_token_rows: tuple[tuple[int, ...], ...] = ()
    """Token rows hashed for TF-IDF similarity (multimodal text stream)."""

    audio_token_rows: tuple[tuple[int, ...], ...] = ()
    """Token rows hashed for TF-IDF similarity (multimodal audio stream)."""

    logprobs: tuple[float, ...] | None = None
    """Natural-log probability of each generated token; ``None`` when the
    backend did not return them."""

    embedding: tuple[float, ...] = ()
    """Static embedding vector for embedding-based value functions."""

    contextual_embedding: tuple[float, ...] = ()
    """Contextual embedding vector for U2-style scoring."""
