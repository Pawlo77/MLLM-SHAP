"""Logprob-based value function.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_logprob.py]]
"""

from typing import TYPE_CHECKING, ClassVar

from ..domain.generation import GenerationRecord
from ..errors import LogprobsUnavailableError
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class LogprobValue:
    """Score a coalition by the model's confidence in its own generation.

    ``v(S)`` is the mean natural-log probability of the tokens the backend
    generated for coalition ``S`` (its ``GenerationRecord.logprobs``), so it
    lies in ``(-inf, 0]``; an empty generation scores 0. The base generation is
    not read: ``v(N)`` is the base generation's own mean logprob.
    """

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "logprob"

    @property
    def needs_logprobs(self) -> bool:
        """Always true; scoring reads ``GenerationRecord.logprobs``."""
        return True

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the mean token logprob of ``candidate``.

        Raises:
            LogprobsUnavailableError: If ``candidate`` carries no logprobs.
        """
        _ = base
        logprobs = (
            candidate.logprobs if isinstance(candidate, GenerationRecord) else None
        )
        if logprobs is None:
            msg = (
                "value_fn 'logprob' needs token logprobs, but the generation "
                f"{candidate.text[:40]!r} has none"
            )
            raise LogprobsUnavailableError(msg)
        return sum(logprobs) / len(logprobs) if logprobs else 0.0
