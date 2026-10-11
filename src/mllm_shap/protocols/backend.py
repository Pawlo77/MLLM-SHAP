"""Generative backend protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from typing import Protocol, runtime_checkable

from ..domain.config import GenerationConfig
from ..domain.conversation import ConversationSnapshot


class GenerationResult(Protocol):
    """Minimal generation output consumed by value functions."""

    @property
    def text(self) -> str:
        """Return the generated text payload."""


class GenerativeBackend(Protocol):
    """Execute model generation for a masked conversation snapshot."""

    @property
    def model_id(self) -> str:
        """Return the backend model identifier."""

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationResult:
        """Generate model output for the given snapshot."""


@runtime_checkable
class SupportsKvCache(Protocol):
    """Backend that can reuse prompt-prefix KV state across coalitions."""

    @property
    def kv_cache_hits(self) -> int:
        """Return the cumulative number of prefix-cache hits."""

    @property
    def kv_cache_reused_tokens(self) -> int:
        """Return the cumulative number of prompt tokens served from the cache."""

    def set_kv_cache_enabled(self, enabled: bool, max_entries: int) -> None:
        """Turn prefix-cache reuse on or off and cap the number of cached prefixes."""


@runtime_checkable
class SupportsLogprobs(Protocol):
    """Backend whose records carry ``GenerationRecord.logprobs``.

    The ``logprob`` value function requires it; a record that still arrives
    without logprobs (a server that omits them) raises
    :exc:`~mllm_shap.errors.LogprobsUnavailableError` when scored.
    """

    @property
    def returns_logprobs(self) -> bool:
        """Return whether generated records include token logprobs."""


@runtime_checkable
class SupportsAclose(Protocol):
    """Backend holding resources (clients, models) released by ``aclose``."""

    async def aclose(self) -> None:
        """Release backend resources."""
