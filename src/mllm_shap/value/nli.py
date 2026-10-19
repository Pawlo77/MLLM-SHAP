"""Bidirectional entailment log-odds between a coalition answer and the full one.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_semantic.py]]
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo
from .encoders import NliModel, load_nli_model

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

_EPS = 1e-6


class NliOptions(PluginOptions):
    """Options of the ``nli_logodds`` value function."""

    model: str = "microsoft/deberta-large-mnli"
    """Hugging Face natural-language-inference model."""

    question: str | None = None
    """Prepended to both answers so entailment is judged in context."""

    revision: str = "main"
    """Hub revision of the model; pin a commit to reproduce runs."""


class NliLogOddsValue:
    r"""Score the log-odds that the two answers entail each other.

    With :math:`p_\to` the probability that :math:`y_N` entails :math:`y_S` and
    :math:`p_\leftarrow` the reverse, :math:`p = \sqrt{p_\to p_\leftarrow}` and
    :math:`v(S) = \log\frac{p}{1-p}`, as the Log NLI score of MExGen. A negated
    answer scores low where similarity scores stay high. The pipeline calls
    :meth:`fit` so all pairs of an instance are scored in batches.
    """

    options_model: ClassVar[type[PluginOptions]] = NliOptions
    """Options accepted from ``explanation.plugin_options``."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(math.log(_EPS), -math.log(_EPS)),
        reads_base=True,
        extra_model="Hugging Face NLI model (default deberta-large-mnli)",
        cost="encoder",
    )

    def __init__(
        self, context: "PluginContext | None" = None, model: NliModel | None = None
    ) -> None:
        """Read the options in ``context``; ``model`` injects a loaded NLI model."""
        options = NliOptions()
        if context is not None:
            options = NliOptions.model_validate(context.options)
        self._options = options
        self._model = model
        self._entail: dict[tuple[str, str], float] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "nli_logodds"

    def manifest_info(self) -> dict[str, str]:
        """Return the NLI model and question for the run manifest."""
        return {
            "model": self._options.model,
            "revision": self._options.revision,
            "question": str(self._options.question),
        }

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        """Score every answer pair of the run in batches."""
        base = corpus[0].text
        self._entail_all([(base, record.text) for record in corpus])

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the log-odds of mutual entailment."""
        self._entail_all([(base.text, candidate.text)])
        forward = self._entail[base.text, candidate.text]
        backward = self._entail[candidate.text, base.text]
        p = min(max(math.sqrt(forward * backward), _EPS), 1.0 - _EPS)
        return math.log(p / (1.0 - p))

    def _entail_all(self, pairs: Sequence[tuple[str, str]]) -> None:
        wanted = {
            key for a, b in pairs for key in ((a, b), (b, a)) if key not in self._entail
        }
        if not wanted:
            return
        if self._model is None:
            self._model = load_nli_model(self._options.model, self._options.revision)
        ordered = sorted(wanted)
        prefix = f"{self._options.question} " if self._options.question else ""
        probs = self._model.entailment([(prefix + a, prefix + b) for a, b in ordered])
        self._entail.update(zip(ordered, probs, strict=True))
