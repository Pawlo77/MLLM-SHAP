"""Protocol contracts for pipeline plugins.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from .backend import (
    GenerationResult,
    GenerativeBackend,
    SupportsAclose,
    SupportsKvCache,
    SupportsLogprobs,
)
from .embedding import EmbeddingProvider
from .estimand import EstimandAggregator
from .estimator import EstimatorStrategy, TwoPhaseEstimator
from .masking import AbsencePolicy
from .partition import PlayerPartitioner
from .plugin import DeclaresPluginOptions, PluginOptions
from .value import (
    FittableValueFunction,
    RequiresEmbeddings,
    RequiresLogprobs,
    ValueFunction,
)

__all__ = [
    "AbsencePolicy",
    "DeclaresPluginOptions",
    "EmbeddingProvider",
    "EstimandAggregator",
    "EstimatorStrategy",
    "FittableValueFunction",
    "GenerationResult",
    "GenerativeBackend",
    "PlayerPartitioner",
    "PluginOptions",
    "RequiresEmbeddings",
    "RequiresLogprobs",
    "SupportsAclose",
    "SupportsKvCache",
    "SupportsLogprobs",
    "TwoPhaseEstimator",
    "ValueFunction",
]
