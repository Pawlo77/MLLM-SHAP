"""Generative backend protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..domain.config import GenerationConfig
from ..domain.conversation import ConversationSnapshot
from ..domain.generation import ForcedScore
from ..runtime.metrics import CostCounters

if TYPE_CHECKING:
    from ..runtime.archive import CoalitionRecord


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
class ReportsCost(Protocol):
    """Backend that counts its model work."""

    @property
    def cost_counters(self) -> CostCounters:
        """Return the cumulative counters since the backend was created."""


@runtime_checkable
class ReplaySource(Protocol):
    """Backend that answers coalitions from archived rows."""

    @property
    def has_fallback(self) -> bool:
        """Return whether a live backend computes coalitions that are not archived."""

    def archived_row(self, coalition_key: str) -> "CoalitionRecord | None":
        """Return the archived row of a coalition key, if any."""

    def archived_base(
        self, snapshot_id: str
    ) -> "tuple[str, CoalitionRecord | None] | None":
        """Return the archived base text and grand-coalition row of a snapshot."""


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
class SupportsTeacherForcing(Protocol):
    """Backend that scores fixed continuations under a prompt without generating.

    Value functions that declare ``needs_teacher_forcing`` require it. ``mock``
    and ``transformers`` (text prompts) implement it; ``lmstudio`` and ``api``
    do not, because chat servers do not return prompt logprobs.
    """

    @property
    def forces_continuations(self) -> bool:
        """Return whether :meth:`score_continuations` is available."""

    async def score_continuations(
        self,
        snapshot: ConversationSnapshot,
        continuations: Sequence[str],
        top_k: int = 0,
    ) -> list[ForcedScore]:
        """Return one :class:`~mllm_shap.domain.generation.ForcedScore` each.

        With ``top_k`` above zero each score also holds the ``top_k`` most
        probable next tokens at every continuation position.
        """


@runtime_checkable
class SupportsAclose(Protocol):
    """Backend holding resources (clients, models) released by ``aclose``."""

    async def aclose(self) -> None:
        """Release backend resources."""
