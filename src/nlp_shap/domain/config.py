"""Generation and runtime settings shared by backends, runtime and pipeline.

@domain domain
@see [[knowledge/architecture/domain]]
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GenerationConfig(BaseModel):
    """Generation parameters for the base and coalition evaluations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: int = 128
    """Maximum tokens generated per coalition evaluation."""

    temperature: float = 0.0
    """Sampling temperature for model generation."""

    top_k: int = 1
    """Top-k sampling cutoff for model generation."""

    compute_embeddings: bool = False
    """Ask backends that can (transformers) to attach model embeddings to records.

    The pipeline turns it on when the value function declares
    ``needs_embeddings``; it costs extra forward passes per coalition.
    """


class DedupConfig(BaseModel):
    """Coalition deduplication options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Literal["auto", "on", "off"] = "auto"
    """Deduplication mode for repeated coalition evaluations."""


class KvCacheConfig(BaseModel):
    """Prefix cache options for supported backends."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    """Whether prefix/KV cache reuse is enabled when supported."""

    max_entries: int = Field(default=8, ge=1)
    """Maximum prefix KV states a backend keeps; the oldest is evicted first."""
