"""Coalition utility scoring plugins.

@domain value
@see [[knowledge/architecture/value]]
"""

from .embedding import CosineEmbeddingValue, EuclideanEmbeddingValue
from .logprob import LogprobValue
from .tfidf import TfIdfCosineValue
from .wer import ReferenceWerValue, word_error_rate

__all__ = [
    "CosineEmbeddingValue",
    "EuclideanEmbeddingValue",
    "LogprobValue",
    "ReferenceWerValue",
    "TfIdfCosineValue",
    "word_error_rate",
]
