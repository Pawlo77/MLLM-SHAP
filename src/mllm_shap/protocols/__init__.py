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
    SupportsTeacherForcing,
)
from .embedding import EmbeddingProvider
from .estimand import EstimandAggregator
from .estimator import EstimatorStrategy, TwoPhaseEstimator
from .masking import AbsencePolicy
from .partition import PlayerPartitioner
from .plugin import DeclaresPluginOptions, PluginOptions
from .value import (
    FittableValueFunction,
    PreparesScores,
    RequiresEmbeddings,
    RequiresLogprobs,
    RequiresTeacherForcing,
    ValueFunction,
    ValueInfo,
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
    "PreparesScores",
    "RequiresEmbeddings",
    "RequiresLogprobs",
    "RequiresTeacherForcing",
    "SupportsAclose",
    "SupportsKvCache",
    "SupportsLogprobs",
    "SupportsTeacherForcing",
    "TwoPhaseEstimator",
    "ValueFunction",
    "ValueInfo",
]
