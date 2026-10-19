"""Contrastive log-odds of a target answer against a foil.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_teacher_forced.py]]
"""

import math
from typing import TYPE_CHECKING, ClassVar

from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo
from .teacher_forced import TeacherForcedValue, forced_for

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class ContrastiveOptions(PluginOptions):
    """Options of the ``contrastive`` value function."""

    target: str
    """Continuation whose probability the explanation raises, such as ``"I can't"``."""

    foil: str
    """Alternative continuation it is weighed against, such as ``"Sure"``."""


class ContrastiveValue(TeacherForcedValue):
    r"""Score :math:`v(S) = \log p(t \mid x_S) - \log p(f \mid x_S)`.

    :math:`t` and :math:`f` are the ``target`` and ``foil`` continuations; each
    probability is the product of its token probabilities under the coalition
    prompt, from one forward pass per continuation. The base generation is not
    read.
    """

    options_model: ClassVar[type[PluginOptions]] = ContrastiveOptions
    """Options accepted from ``explanation.plugin_options``."""

    registered_name = "contrastive"

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-math.inf, math.inf),
        reads_base=False,
        needs=("teacher_forcing",),
        cost="forward",
    )

    def __init__(self, context: "PluginContext") -> None:
        """Read ``target`` and ``foil`` from the plugin options in ``context``."""
        options = ContrastiveOptions.model_validate(context.options)
        self.target = options.target
        """Continuation raised by the explanation."""
        self.foil = options.foil
        """Continuation it is contrasted with."""

    def forced_continuations(self, base: GenerationResult) -> tuple[str, ...]:
        """Return the target and the foil."""
        _ = base
        return (self.target, self.foil)

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the log-odds of the target against the foil."""
        _ = base
        return math.fsum(
            forced_for(candidate, self.target, self.name).token_logprobs
        ) - math.fsum(forced_for(candidate, self.foil, self.name).token_logprobs)
