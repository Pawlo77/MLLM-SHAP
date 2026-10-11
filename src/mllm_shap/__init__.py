"""Shapley-value attributions for conversational language models.

@domain core
@see [[knowledge/architecture/core]]
"""

from importlib.metadata import PackageNotFoundError, version

from ._logging import bootstrap_logging, configure_logging
from .alignment.partitions import SgpaSegmentPartitioner
from .backends.api import ApiBackend
from .backends.lmstudio import LmStudioBackend
from .backends.mock import MockBackend
from .backends.transformers import TransformersTextBackend
from .domain import (
    AudioPart,
    CoalitionMask,
    ConversationSnapshot,
    CooperativeGame,
    Estimand,
    GenerationRecord,
    Message,
    PlayerSet,
    PlayerSpan,
    Role,
    TextPart,
    Turn,
)
from .errors import (
    ArchiveCorruptError,
    ArchiveExistsError,
    ArchiveIncompleteError,
    ArchiveNotFoundError,
    BackendUnavailableError,
    ConfigError,
    LogprobsUnavailableError,
    ManifestVersionError,
    MllmShapError,
    PluginConflictError,
    PluginLoadError,
    PluginNotFoundError,
    ReanalyzeMismatchError,
)
from .estimation.complementary import ComplementaryEstimator
from .estimation.estimands import BanzhafAggregator, ShapleyAggregator
from .estimation.exact import ExactEstimator
from .estimation.monte_carlo import MonteCarloEstimator
from .estimation.neyman import NeymanEstimator
from .estimation.normalizers import (
    AbsSumNormalizer,
    IdentityNormalizer,
    MinMaxNormalizer,
    PowerShiftNormalizer,
)
from .masking import (
    DeletePolicy,
    MessagePartitioner,
    NeutralPolicy,
    PadPolicy,
    TokenPartitioner,
)
from .pipeline import (
    ExplainConfig,
    ExplainResult,
    ExplainRunner,
    ExplainRunOutput,
    RunManifest,
    explain_config_from_yaml,
    explain_config_to_yaml,
    parse_manifest,
)
from .plugins import PluginContext, PluginGroup, PluginRegistry, default_registry
from .runtime import CoalitionProgress, InMemoryObservabilitySink, ObservabilitySink
from .value import (
    CosineEmbeddingValue,
    EuclideanEmbeddingValue,
    LogprobValue,
    TfIdfCosineValue,
)
from .viz.token_bar import TokenBarRenderer
from .viz.token_text import TokenTextRenderer

_LAZY_VIZ = (
    "display_attribution_html",
    "render_attribution",
    "render_attribution_html",
)
"""Exports loaded from :mod:`mllm_shap.viz.render` on first access."""


def __getattr__(name: str) -> object:
    if name in _LAZY_VIZ:
        from .viz import render

        return getattr(render, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AbsSumNormalizer",
    "ApiBackend",
    "ArchiveCorruptError",
    "ArchiveExistsError",
    "ArchiveIncompleteError",
    "ArchiveNotFoundError",
    "AudioPart",
    "BackendUnavailableError",
    "BanzhafAggregator",
    "CoalitionMask",
    "CoalitionProgress",
    "ComplementaryEstimator",
    "ConfigError",
    "ConversationSnapshot",
    "CooperativeGame",
    "CosineEmbeddingValue",
    "DeletePolicy",
    "Estimand",
    "EuclideanEmbeddingValue",
    "ExactEstimator",
    "ExplainConfig",
    "ExplainResult",
    "ExplainRunOutput",
    "ExplainRunner",
    "GenerationRecord",
    "IdentityNormalizer",
    "InMemoryObservabilitySink",
    "LmStudioBackend",
    "LogprobValue",
    "LogprobsUnavailableError",
    "ManifestVersionError",
    "Message",
    "MessagePartitioner",
    "MinMaxNormalizer",
    "MllmShapError",
    "MockBackend",
    "MonteCarloEstimator",
    "NeutralPolicy",
    "NeymanEstimator",
    "ObservabilitySink",
    "PadPolicy",
    "PlayerSet",
    "PlayerSpan",
    "PluginConflictError",
    "PluginContext",
    "PluginGroup",
    "PluginLoadError",
    "PluginNotFoundError",
    "PluginRegistry",
    "PowerShiftNormalizer",
    "ReanalyzeMismatchError",
    "Role",
    "RunManifest",
    "SgpaSegmentPartitioner",
    "ShapleyAggregator",
    "TextPart",
    "TfIdfCosineValue",
    "TokenBarRenderer",
    "TokenPartitioner",
    "TokenTextRenderer",
    "TransformersTextBackend",
    "Turn",
    "__version__",
    "configure_logging",
    "default_registry",
    "display_attribution_html",
    "explain_config_from_yaml",
    "explain_config_to_yaml",
    "parse_manifest",
    "render_attribution",
    "render_attribution_html",
]

try:
    __version__ = version("mllm-shap")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

bootstrap_logging()
