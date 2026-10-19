"""BERTScore F1 between a coalition answer and the full-input answer.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_semantic.py]]
"""

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo
from .encoders import TokenEncoder, load_token_encoder

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

_BATCH = 32


class BertScoreOptions(PluginOptions):
    """Options of the ``bertscore`` value function."""

    encoder: str = "roberta-large"
    """Hugging Face encoder whose hidden states are matched."""

    layer: int = 17
    """Hidden layer used (17 is the reference package's choice for roberta-large)."""

    baseline: float | None = None
    """Baseline F1 of unrelated pairs; the score becomes ``(F1 - b) / (1 - b)``."""

    revision: str = "main"
    """Hub revision of the model; pin a commit to reproduce runs."""


class BertScoreValue:
    r"""Score :math:`v(S) = F_1` of greedy token matching against :math:`y_N`.

    Recall averages, over :math:`y_N` tokens, the best cosine similarity to a
    token of :math:`y_S`; precision does the reverse; :math:`F_1` is their
    harmonic mean (Zhang et al., 2020). Special tokens are dropped and no IDF
    weighting is applied, as in the reference package's default. The pipeline
    calls :meth:`fit` so all answers of an instance are encoded in batches.
    """

    options_model: ClassVar[type[PluginOptions]] = BertScoreOptions
    """Options accepted from ``explanation.plugin_options``."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-1.0, 1.0),
        reads_base=True,
        extra_model="Hugging Face encoder (default roberta-large)",
        cost="encoder",
    )

    def __init__(
        self,
        context: "PluginContext | None" = None,
        encoder: TokenEncoder | None = None,
    ) -> None:
        """Read the options in ``context``; ``encoder`` injects a loaded encoder."""
        options = BertScoreOptions()
        if context is not None:
            options = BertScoreOptions.model_validate(context.options)
        self._options = options
        self._encoder = encoder
        self._vectors: dict[str, NDArray[np.floating]] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "bertscore"

    def manifest_info(self) -> dict[str, str]:
        """Return encoder, layer, baseline and library version for the manifest."""
        try:
            library = version("transformers")
        except PackageNotFoundError:
            library = "unknown"
        return {
            "encoder": self._options.encoder,
            "layer": str(self._options.layer),
            "revision": self._options.revision,
            "baseline": str(self._options.baseline),
            "transformers": library,
        }

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        """Encode every distinct answer of the run in batches."""
        self._encode_all([record.text for record in corpus])

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the (optionally rescaled) BERTScore F1 of ``candidate``."""
        self._encode_all([base.text, candidate.text])
        reference = self._vectors[base.text]
        hypothesis = self._vectors[candidate.text]
        if not len(reference) or not len(hypothesis):
            return 0.0
        similarity = reference @ hypothesis.T
        recall = float(similarity.max(axis=1).mean())
        precision = float(similarity.max(axis=0).mean())
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        baseline = self._options.baseline
        return f1 if baseline is None else (f1 - baseline) / (1.0 - baseline)

    def _encode_all(self, texts: Sequence[str]) -> None:
        missing = sorted({t for t in texts if t not in self._vectors})
        if not missing:
            return
        if self._encoder is None:
            self._encoder = load_token_encoder(
                self._options.encoder, self._options.layer, self._options.revision
            )
        for start in range(0, len(missing), _BATCH):
            chunk = missing[start : start + _BATCH]
            self._vectors.update(zip(chunk, self._encoder.encode(chunk), strict=True))
