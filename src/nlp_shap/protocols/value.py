"""Value function protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ..domain.generation import GenerationRecord
from .backend import GenerationResult


class ValueFunction(Protocol):
    """Score coalition utility relative to a base generation."""

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return utility for ``candidate`` relative to ``base``."""


@runtime_checkable
class FittableValueFunction(ValueFunction, Protocol):
    """Value function that must see the run's corpus before it scores.

    The pipeline calls :meth:`fit` with the base generation and every coalition
    generation before any :meth:`~ValueFunction.score` call, in ``explain`` and
    ``reanalyze``.
    """

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        """Freeze corpus statistics used by :meth:`~ValueFunction.score`."""


@runtime_checkable
class RequiresLogprobs(Protocol):
    """Value function that scores ``GenerationRecord.logprobs``.

    When :attr:`needs_logprobs` is true the pipeline refuses backends that are
    not :class:`~nlp_shap.protocols.backend.SupportsLogprobs` and ``reanalyze``
    refuses archives whose rows have no logprobs.
    """

    @property
    def needs_logprobs(self) -> bool:
        """Return whether scoring reads ``logprobs``."""


@runtime_checkable
class RequiresEmbeddings(Protocol):
    """Value function that scores model embeddings rather than text.

    When :attr:`needs_embeddings` is true the pipeline sets
    ``GenerationConfig.compute_embeddings`` for its backend calls, so backends
    that can (transformers) attach static and contextual embeddings to their
    records, and scores those records.
    """

    @property
    def needs_embeddings(self) -> bool:
        """Return whether scoring reads ``embedding`` or ``contextual_embedding``."""
