"""Frozen-corpus TF-IDF cosine value function.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_tfidf.py]]
"""

import math
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo
from ._metrics import cosine_similarity
from ._token_hash import TokenHasher

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class TfIdfCosineValue:
    """Score generations with cosine similarity of frozen TF-IDF vectors."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(0.0, 1.0), reads_base=True, cost="text"
    )
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context
        self._hasher = TokenHasher()
        self._idf: dict[int, float] | None = None
        self._fitted = False

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "tfidf_cosine"

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        """Freeze inverse document frequency weights from ``corpus``."""
        if not corpus:
            msg = "corpus must contain at least one generation"
            raise ValueError(msg)
        documents = [self._hasher.document_token_ids(record) for record in corpus]
        document_count = len(documents)
        document_frequency: Counter[int] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        self._idf = {
            term: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._fitted = True

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return TF-IDF cosine similarity between ``candidate`` and ``base``."""
        if not self._fitted or self._idf is None:
            msg = "TfIdfCosineValue.fit must be called before scoring"
            raise RuntimeError(msg)
        base_record = _require_generation_record(base)
        candidate_record = _require_generation_record(candidate)
        base_vector = self._tfidf_vector(self._hasher.document_token_ids(base_record))
        candidate_vector = self._tfidf_vector(
            self._hasher.document_token_ids(candidate_record)
        )
        base_dense, candidate_dense = _dict_to_dense(base_vector, candidate_vector)
        return cosine_similarity(base_dense, candidate_dense)

    def _tfidf_vector(self, token_ids: Sequence[int]) -> dict[int, float]:
        idf = self._idf
        if idf is None:
            msg = "TF-IDF weights are not initialized"
            raise RuntimeError(msg)
        term_frequency = Counter(token_ids)
        if not term_frequency:
            return {}
        max_frequency = max(term_frequency.values())
        vector: dict[int, float] = {}
        for term, count in term_frequency.items():
            tf = count / max_frequency
            vector[term] = tf * idf.get(term, 1.0)
        return vector


def _require_generation_record(generation: GenerationResult) -> GenerationRecord:
    if not isinstance(generation, GenerationRecord):
        msg = "TfIdfCosineValue requires GenerationRecord token rows"
        raise TypeError(msg)
    return generation


def _dict_to_dense(
    left: dict[int, float],
    right: dict[int, float],
) -> tuple[list[float], list[float]]:
    vocabulary = sorted(set(left) | set(right))
    left_dense = [left.get(term, 0.0) for term in vocabulary]
    right_dense = [right.get(term, 0.0) for term in vocabulary]
    return left_dense, right_dense
