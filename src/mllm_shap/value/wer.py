"""Word error rate against a reference text as a coalition value.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_wer.py]]
"""

import re
from typing import TYPE_CHECKING, ClassVar

from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

_WORD_RE = re.compile(r"[\w']+")


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word-level edit distance over the reference's word count.

    Words are lowercased runs of letters, digits, ``_`` and ``'``, so case and
    punctuation do not count. An empty reference gives 0 for an empty
    hypothesis and 1 otherwise.
    """
    ref = _WORD_RE.findall(reference.lower())
    hyp = _WORD_RE.findall(hypothesis.lower())
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, start=1):
        current = [i]
        for j, other in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (word != other),
                )
            )
        previous = current
    return previous[-1] / len(ref)


class ReferenceWerOptions(PluginOptions):
    """Options of the ``reference_wer`` value function."""

    reference: str | None = None
    """Target text, such as a transcript; ``None`` uses the full coalition's answer."""


class ReferenceWerValue:
    """Score ``1 - min(1, WER)`` of a coalition's answer against a reference.

    With ``reference`` set, ``v(S)`` measures how correct the answer is (a
    transcription task); without it, how close it stays to the full answer.
    Values lie in ``[0, 1]``.
    """

    options_model: ClassVar[type[PluginOptions]] = ReferenceWerOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read ``reference`` from the plugin options in ``context``."""
        options = ReferenceWerOptions()
        if context is not None:
            options = ReferenceWerOptions.model_validate(context.options)
        self.reference = options.reference
        """Target text; ``None`` scores against the base generation."""

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "reference_wer"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return ``1 - min(1, WER)`` of ``candidate`` against the target."""
        target = base.text if self.reference is None else self.reference
        return 1.0 - min(1.0, word_error_rate(target, candidate.text))
