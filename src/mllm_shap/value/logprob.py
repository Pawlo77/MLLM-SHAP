"""Self-confidence value function from generated-token logprobs.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_logprob.py]]
"""

import math
import warnings
from typing import TYPE_CHECKING, ClassVar

from ..domain.generation import GenerationRecord
from ..errors import LogprobsUnavailableError
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class SelfLogprobValue:
    """Score a coalition by the model's confidence in its own generation.

    ``v(S)`` is the mean natural-log probability of the tokens the backend
    generated for coalition ``S`` (its ``GenerationRecord.logprobs``), so it
    lies in ``(-inf, 0]``; an empty generation scores 0. The base generation is
    not read: ``v(N)`` is the base generation's own mean logprob.
    """

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    registered_name: ClassVar[str] = "self_logprob"
    """Registered value-function identifier."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-math.inf, 0.0),
        reads_base=False,
        needs=("logprobs",),
        cost="text",
    )

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return self.registered_name

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
                f"value_fn {self.name!r} needs token logprobs, but the generation "
                f"{candidate.text[:40]!r} has none"
            )
            raise LogprobsUnavailableError(msg)
        return sum(logprobs) / len(logprobs) if logprobs else 0.0


class LogprobValue(SelfLogprobValue):
    """Deprecated name of :class:`SelfLogprobValue`, kept for one minor version.

    It measures confidence in the coalition's own answer and ignores :math:`y_N`;
    use ``tf_logprob`` for fidelity to the full-input answer.
    """

    registered_name = "logprob"

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Warn that ``logprob`` is a confidence score and point to ``tf_logprob``."""
        warnings.warn(
            "value_fn 'logprob' is renamed 'self_logprob' (confidence in the "
            "coalition's own answer, ignoring the full-input answer); "
            "use 'tf_logprob' for teacher-forced fidelity",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(context)
