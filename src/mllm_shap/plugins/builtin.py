"""Built-in plugin registration and the default registry.

@domain plugins
@see [[knowledge/architecture/plugins]]
@tested-by [[tests/plugins/test_builtin.py]]
"""

from ..alignment.partitions import SgpaSegmentPartitioner
from ..backends.api import ApiBackend
from ..backends.lmstudio import LmStudioBackend
from ..backends.mock import MockAudioBackend, MockBackend
from ..backends.transformers import TransformersTextBackend
from ..estimation.complementary import ComplementaryEstimator
from ..estimation.estimands import BanzhafAggregator, ShapleyAggregator
from ..estimation.exact import ExactEstimator
from ..estimation.kernel_banzhaf import KernelBanzhafEstimator
from ..estimation.leverage import LeverageEstimator
from ..estimation.loo import LeaveOneOutEstimator
from ..estimation.monte_carlo import MonteCarloEstimator
from ..estimation.neyman import NeymanEstimator
from ..estimation.normalizers import (
    AbsSumNormalizer,
    IdentityNormalizer,
    MinMaxNormalizer,
    PowerShiftNormalizer,
)
from ..estimation.proxy import ProxyEstimator
from ..masking.partitions import (
    MessagePartitioner,
    SentencePartitioner,
    TokenPartitioner,
)
from ..masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from ..value.bertscore import BertScoreValue
from ..value.contrastive import ContrastiveValue
from ..value.embedding import CosineEmbeddingValue
from ..value.judge import JudgeGradeValue
from ..value.logprob import LogprobValue, SelfLogprobValue
from ..value.nli import NliLogOddsValue
from ..value.semascore import SemaScoreValue
from ..value.speech import SpeechBertScoreValue, TranscribedValue
from ..value.teacher_forced import TfJsdValue, TfLogprobValue
from ..value.tfidf import TfIdfCosineValue
from ..value.wer import ReferenceWerValue
from ..viz.token_bar import TokenBarRenderer
from ..viz.token_text import TokenTextRenderer
from .groups import PluginGroup
from .registry import PluginFactory, PluginRegistry

BUILTIN_PLUGINS: dict[PluginGroup, dict[str, PluginFactory]] = {
    PluginGroup.ESTIMATORS: {
        "exact": ExactEstimator,
        "mc": MonteCarloEstimator,
        "complementary": ComplementaryEstimator,
        "neyman_cc": NeymanEstimator,
        "leverage": LeverageEstimator,
        "kernel_banzhaf": KernelBanzhafEstimator,
        "proxy": ProxyEstimator,
        "loo": LeaveOneOutEstimator,
    },
    PluginGroup.ESTIMANDS: {
        "shapley": ShapleyAggregator,
        "banzhaf": BanzhafAggregator,
    },
    PluginGroup.VALUE_FNS: {
        "tfidf_cosine": TfIdfCosineValue,
        "embedding_cosine": CosineEmbeddingValue,
        "self_logprob": SelfLogprobValue,
        "logprob": LogprobValue,
        "tf_logprob": TfLogprobValue,
        "tf_jsd": TfJsdValue,
        "contrastive": ContrastiveValue,
        "bertscore": BertScoreValue,
        "nli_logodds": NliLogOddsValue,
        "semascore": SemaScoreValue,
        "speech_bertscore": SpeechBertScoreValue,
        "transcribed": TranscribedValue,
        "judge_grade": JudgeGradeValue,
        "reference_wer": ReferenceWerValue,
    },
    PluginGroup.NORMALIZERS: {
        "identity": IdentityNormalizer,
        "abs_sum": AbsSumNormalizer,
        "power_shift": PowerShiftNormalizer,
        "min_max": MinMaxNormalizer,
    },
    PluginGroup.BACKENDS: {
        "mock": MockBackend,
        "mock_audio": MockAudioBackend,
        "lmstudio": LmStudioBackend,
        "transformers": TransformersTextBackend,
        "api": ApiBackend,
    },
    PluginGroup.PARTITIONS: {
        "tokens": TokenPartitioner,
        "sentences": SentencePartitioner,
        "messages": MessagePartitioner,
        "sgpa_segments": SgpaSegmentPartitioner,
    },
    PluginGroup.ABSENCE_POLICIES: {
        "delete": DeletePolicy,
        "pad": PadPolicy,
        "neutral": NeutralPolicy,
    },
    PluginGroup.RENDERERS: {
        "token_text": TokenTextRenderer,
        "token_bar": TokenBarRenderer,
    },
}
"""Built-in plugin factories per group; mirrors the ``pyproject.toml`` entry points."""


def register_builtin_plugins(registry: PluginRegistry) -> None:
    """Register every built-in plugin of all eight groups in ``registry``."""
    for group, factories in BUILTIN_PLUGINS.items():
        for name, factory in factories.items():
            registry.register(group, name, factory)


def default_registry() -> PluginRegistry:
    """Return a registry with the built-in plugins and third-party entry points.

    Entry points that reuse a built-in name are skipped.

    Raises:
        PluginLoadError: If a third-party entry point cannot be imported.
    """
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    for group in PluginGroup:
        registry.load_entry_points(group)
    return registry
