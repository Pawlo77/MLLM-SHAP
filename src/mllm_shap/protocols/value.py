"""Value function protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain.generation import GenerationRecord
from .backend import GenerationResult


@dataclass(frozen=True, slots=True)
class ValueInfo:
    """What a value function measures and costs; every built-in declares one.

    Declared as the class attribute ``info``; ``docs/theory/value_functions.rst``
    lists it and a test checks every built-in has it.
    """

    value_range: tuple[float, float]
    """Lowest and highest possible score (``-inf`` / ``inf`` when unbounded)."""

    reads_base: bool
    """Whether the score depends on the full-input answer :math:`y_N`."""

    needs: tuple[str, ...] = ()
    """Capabilities required of the backend or archive, from ``logprobs``,
    ``embeddings``, ``teacher_forcing``, ``judge`` and ``audio``."""

    extra_model: str = ""
    """Model the value function loads besides the explained one; empty for none."""

    cost: str = "text"
    """Cost per coalition, one of ``text`` (string work), ``encoder`` (one small
    encoder pass), ``forward`` (one forward pass of the explained model) and
    ``judge`` (a judge model call)."""

    higher_is_better: bool = True
    """Whether a larger score means a coalition answer closer to the full one."""


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
    not :class:`~mllm_shap.protocols.backend.SupportsLogprobs` and ``reanalyze``
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


@runtime_checkable
class RequiresTeacherForcing(Protocol):
    """Value function that scores teacher-forced continuations.

    When :attr:`needs_teacher_forcing` is true the pipeline refuses backends that
    are not :class:`~mllm_shap.protocols.backend.SupportsTeacherForcing`, scores
    :meth:`forced_continuations` under every coalition prompt and attaches the
    results to ``GenerationRecord.forced``. With :attr:`reads_candidate_text`
    false it also skips generating the coalition answers.
    """

    @property
    def needs_teacher_forcing(self) -> bool:
        """Return whether scoring reads ``GenerationRecord.forced``."""

    @property
    def forced_top_k(self) -> int:
        """Return how many next-token candidates per position to request."""

    @property
    def reads_candidate_text(self) -> bool:
        """Return whether scoring reads the coalition's generated text."""

    def forced_continuations(self, base: GenerationResult) -> tuple[str, ...]:
        """Return the continuations to score under each coalition prompt."""


@runtime_checkable
class PreparesScores(Protocol):
    """Value function that needs asynchronous work, such as a judge model call.

    The pipeline awaits :meth:`prepare` with the base generation and every
    coalition generation before any ``score`` call.
    """

    async def prepare(
        self, base: GenerationRecord, corpus: Sequence[GenerationRecord]
    ) -> None:
        """Compute and cache what ``score`` reads."""
