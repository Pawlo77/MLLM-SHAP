"""SeMaScore: segment-aligned, importance-weighted meaning score for transcripts.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_semantic.py]]
"""

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo
from ._metrics import cosine_similarity
from .encoders import WordEncoder, load_word_encoder

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

_WORD_RE = re.compile(r"[\w']+")
_BATCH = 32


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def align_segments(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Align two word lists into segment pairs of half-open word spans.

    A word-level edit-distance path is cut into matches (one word on each side)
    and runs of consecutive errors, which form one segment so a split or merged
    word (two words against one) is compared as a unit. A run that
    inserts or deletes only has an empty span on one side.
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
            )
    steps: list[tuple[int, int, int, int, bool]] = []
    i, j = n, m
    while i or j:
        if (
            i
            and j
            and cost[i][j]
            == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
        ):
            steps.append((i - 1, i, j - 1, j, reference[i - 1] == hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            steps.append((i - 1, i, j, j, False))
            i -= 1
        else:
            steps.append((i, i, j - 1, j, False))
            j -= 1
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    run: list[tuple[int, int, int, int, bool]] = []
    for step in reversed(steps):
        if step[4]:
            if run:
                segments.append(_merge(run))
                run = []
            segments.append(((step[0], step[1]), (step[2], step[3])))
        else:
            run.append(step)
    if run:
        segments.append(_merge(run))
    return segments


def _merge(
    run: list[tuple[int, int, int, int, bool]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (run[0][0], run[-1][1]), (run[0][2], run[-1][3])


class SemaScoreOptions(PluginOptions):
    """Options of the ``semascore`` value function."""

    reference: str | None = None
    """Target transcript; ``None`` scores against the full-input answer."""

    encoder: str = "roberta-base"
    """Hugging Face encoder of contextual word vectors."""

    revision: str = "main"
    """Hub revision of the model; pin a commit to reproduce runs."""


class SemaScoreValue:
    r"""Score a transcript by aligned segments weighted by their importance.

    For each aligned segment pair :math:`(g_i, h_i)` the segment score is
    :math:`\cos(e_{g_i}, e_{h_i})(1 - \mathrm{MER}_i)`, with :math:`\mathrm{MER}_i`
    the character edit distance over the longer segment, and its weight is
    :math:`\alpha_i = \max(0, \cos(e_{g_i}, e_y))`, the segment's closeness to the
    whole reference (Sasindran et al., 2024). :math:`v(S) = \sum_i \alpha_i
    \mathrm{SS}_i / \sum_i \alpha_i`. Word vectors are the mean of the sub-word
    states of the encoder's last layer. An empty reference scores 1 for an empty
    hypothesis and 0 otherwise.
    """

    options_model: ClassVar[type[PluginOptions]] = SemaScoreOptions
    """Options accepted from ``explanation.plugin_options``."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(0.0, 1.0),
        reads_base=True,
        extra_model="Hugging Face encoder (default roberta-base)",
        cost="encoder",
    )

    def __init__(
        self,
        context: "PluginContext | None" = None,
        encoder: WordEncoder | None = None,
    ) -> None:
        """Read the options in ``context``; ``encoder`` injects a loaded encoder."""
        options = SemaScoreOptions()
        if context is not None:
            options = SemaScoreOptions.model_validate(context.options)
        self._options = options
        self._encoder = encoder
        self._vectors: dict[tuple[str, ...], NDArray[np.floating]] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "semascore"

    def manifest_info(self) -> dict[str, str]:
        """Return the encoder for the run manifest."""
        return {
            "encoder": self._options.encoder,
            "revision": self._options.revision,
        }

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        """Encode every distinct answer of the run in batches."""
        self._encode_all([_words(record.text) for record in corpus])

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the importance-weighted segment score of ``candidate``."""
        target = (
            base.text if self._options.reference is None else self._options.reference
        )
        reference, hypothesis = _words(target), _words(candidate.text)
        if not reference:
            return 1.0 if not hypothesis else 0.0
        self._encode_all([reference, hypothesis])
        ref_vectors = self._vectors[tuple(reference)]
        hyp_vectors = self._vectors[tuple(hypothesis)] if hypothesis else None
        whole = ref_vectors.mean(axis=0)
        numerator = denominator = 0.0
        for (g0, g1), (h0, h1) in align_segments(reference, hypothesis):
            weight = (
                max(0.0, cosine_similarity(ref_vectors[g0:g1].mean(axis=0), whole))
                if g1 > g0
                else 0.0
            )
            if g1 == g0 or h1 == h0 or hyp_vectors is None:
                segment = 0.0
            else:
                gold, heard = " ".join(reference[g0:g1]), " ".join(hypothesis[h0:h1])
                mer = _edit_distance(gold, heard) / max(len(gold), len(heard))
                similarity = cosine_similarity(
                    ref_vectors[g0:g1].mean(axis=0), hyp_vectors[h0:h1].mean(axis=0)
                )
                segment = max(0.0, similarity) * (1.0 - mer)
            numerator += weight * segment
            denominator += weight
        return numerator / denominator if denominator else 0.0

    def _encode_all(self, sentences: Sequence[Sequence[str]]) -> None:
        missing = sorted({tuple(s) for s in sentences if s} - self._vectors.keys())
        if not missing:
            return
        if self._encoder is None:
            self._encoder = load_word_encoder(
                self._options.encoder, self._options.revision
            )
        for start in range(0, len(missing), _BATCH):
            chunk = missing[start : start + _BATCH]
            self._vectors.update(zip(chunk, self._encoder.encode(chunk), strict=True))
