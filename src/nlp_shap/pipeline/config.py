"""Explain pipeline configuration schema.

``GenerationConfig``, ``DedupConfig`` and ``KvCacheConfig`` live in
:mod:`nlp_shap.domain.config` so the runtime can use them without importing the
pipeline; they are re-exported here.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_config.py]]
"""

from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
    model_validator,
)

from ..domain.config import DedupConfig, GenerationConfig, KvCacheConfig
from ..domain.enums import EmbeddingMode
from ..domain.estimands import Estimand
from ..errors import ConfigError

__all__ = [
    "ArchiveConfig",
    "BackendConfig",
    "BudgetConfig",
    "DedupConfig",
    "ExplainConfig",
    "ExplanationConfig",
    "GenerationConfig",
    "KvCacheConfig",
    "NeymanConfig",
    "explain_config_from_yaml",
    "explain_config_to_yaml",
]


class BackendConfig(BaseModel):
    """Backend selection and connection parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    """Registered backend plugin identifier."""

    model_id: str
    """Model name or repository id passed to the backend."""

    api_host: str | None = None
    """API host for HTTP-compatible backends.

    Required for the ``api`` backend (there is no environment or localhost
    fallback); optional for ``lmstudio``, which otherwise asks LM Studio for its
    local server. The ``api`` backend sends ``NLP_SHAP_API_KEY`` as the bearer key
    to any host; ``OPENAI_API_KEY`` is used only when that is unset and the host
    is ``api.openai.com``.
    """

    timeout_s: float = Field(default=120.0, gt=0.0)
    """Per-request timeout in seconds for remote backends."""

    max_retries: int = Field(default=3, ge=0)
    """Retries after a transient backend failure (0 disables retrying)."""

    options: dict[str, object] = {}
    """Backend-specific settings validated by the selected backend."""

    @model_validator(mode="after")
    def _api_host_required(self) -> Self:
        if self.kind == "api" and not self.api_host:
            msg = (
                "backend.api_host is required for the api backend "
                "(for example https://api.openai.com/v1 or http://127.0.0.1:1234/v1)"
            )
            raise ConfigError(msg)
        return self


class BudgetConfig(BaseModel):
    """Coalition budget: a fraction of the ``2^n`` coalitions or an absolute count.

    Set at most one of ``fraction`` and ``evaluations``; with neither the budget
    is the full table (``fraction: 1.0``). Only the key that is set is
    serialised, so configurations without ``evaluations`` keep their run id.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    """Share of the ``2^n`` coalitions evaluated; 1.0 when ``evaluations`` is unset."""

    evaluations: int | None = Field(default=None, ge=1, strict=True)
    """Coalition evaluations in total, empty and grand included, instead of
    ``fraction``; raised to the estimator's floor (2 for ``mc``, ``2n + 2`` for
    ``complementary`` and ``neyman_cc``, which also round down to whole pairs)
    and capped at ``2^n``."""

    @model_validator(mode="before")
    @classmethod
    def _one_budget(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fraction = data.get("fraction")
        evaluations = data.get("evaluations")
        if fraction is not None and evaluations is not None:
            msg = (
                "set exactly one of budget.fraction and budget.evaluations, "
                f"got fraction {fraction} and evaluations {evaluations}"
            )
            raise ConfigError(msg)
        if fraction is None and evaluations is None:
            return {**data, "fraction": 1.0}
        return data

    @model_serializer(mode="wrap")
    def _drop_unset(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        dumped: dict[str, Any] = handler(self)
        return {key: value for key, value in dumped.items() if value is not None}


class ArchiveConfig(BaseModel):
    """Run archive persistence options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = "./runs/{run_id}"
    """Filesystem path template for persisted run archives."""

    flush_every: int = Field(default=50, ge=1)
    """Number of coalition records between archive flushes."""

    on_exists: Literal["error", "overwrite"] = "error"
    """What to do when the archive directory already holds a run."""


class NeymanConfig(BaseModel):
    """Neyman-CC estimator controls used when ``estimator`` is ``neyman_cc``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    use_standard_method: bool = False
    """Whether phase-one sampling uses unstructured random coalitions."""

    initial_fraction: float | None = Field(default=None, gt=0, le=1)
    """Fraction of the total budget used during phase-one initialization, in (0, 1]."""

    initial_num_samples: int | None = Field(default=None, ge=1)
    """Explicit phase-one pair budget overriding ``initial_fraction``; at least 1."""


class ExplanationConfig(BaseModel):
    """Explanation algorithm and runtime controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimand: Estimand = Estimand.SHAPLEY
    """Cooperative-game value targeted by aggregation."""

    estimator: str = "neyman_cc"
    """Registered estimator plugin used to sample coalitions."""

    value_fn: str = "tfidf_cosine"
    """Registered value-function plugin used to score outputs."""

    normalizer: str = "identity"
    """Registered normalizer applied after estimand aggregation."""

    players: str = "tokens"
    """Registered partition plugin that defines explainability players."""

    absence_policy: str = "delete"
    """Registered absence policy used to render masked snapshots."""

    budget: BudgetConfig = BudgetConfig()
    """Estimator budget controls."""

    include_minimal_masks: bool = False
    """Whether minimal coalitions are included in estimator sampling."""

    neyman: NeymanConfig = NeymanConfig()
    """Neyman-CC controls applied when the Neyman estimator is selected."""

    max_inflight: int = Field(default=2, ge=1)
    """Maximum concurrent coalition evaluations."""

    max_players: int = Field(default=24, ge=1, le=62)
    """Runs planning more than 2^max_players coalitions (fraction * 2^n, or
    evaluations) are refused. The ceiling of 62 keeps 2^n inside a signed 64-bit
    integer, which the samplers' NumPy mask arithmetic needs."""

    archive: ArchiveConfig = ArchiveConfig()
    """Run archive persistence settings."""

    dedup: DedupConfig = DedupConfig()
    """Coalition deduplication settings."""

    kv_cache: KvCacheConfig = KvCacheConfig()
    """Prefix cache settings for supported backends."""

    embedding_mode: EmbeddingMode = EmbeddingMode.STATIC
    """Embedding mode used by embedding-based value functions."""

    seed: int = 42
    """Random seed for reproducible coalition sampling."""

    plugin_options: dict[str, dict[str, object]] = {}
    """Per-plugin options keyed by plugin name, passed as ``PluginContext.options``."""

    @model_validator(mode="after")
    def _check_estimator_settings(self) -> Self:
        if self.estimator == "exact" and self.budget.fraction != 1.0:
            msg = (
                "estimator 'exact' enumerates every coalition; set budget.fraction: 1.0"
            )
            raise ConfigError(msg)
        if (
            self.estimator in ("complementary", "neyman_cc")
            and self.estimand is not Estimand.SHAPLEY
        ):
            msg = (
                f"estimator {self.estimator!r} estimates the shapley estimand only; "
                "use estimator 'mc' or 'exact' for banzhaf"
            )
            raise ConfigError(msg)
        if self.estimator == "neyman_cc" and self.include_minimal_masks:
            msg = "estimator 'neyman_cc' does not support include_minimal_masks: true"
            raise ConfigError(msg)
        return self


class ExplainConfig(BaseModel):
    """Top-level explain pipeline configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: BackendConfig
    """Backend selection and connection parameters."""

    generation: GenerationConfig = GenerationConfig()
    """Generation parameters for base and coalition evaluations."""

    explanation: ExplanationConfig = ExplanationConfig()
    """Explanation algorithm and runtime controls."""


def explain_config_from_yaml(text: str) -> ExplainConfig:
    """Parse YAML text into a validated :class:`ExplainConfig`.

    Unknown keys and values of the wrong type or range raise pydantic's
    ``ValidationError``.

    Raises:
        ConfigError: If the YAML root is not a mapping (an empty document
            included) or settings are inconsistent with each other (for example
            ``estimator: exact`` with a partial budget).
    """
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        msg = f"explain config root must be a mapping, got {type(data).__name__}"
        raise ConfigError(msg)
    try:
        return ExplainConfig.model_validate(data)
    except ValidationError as error:
        for detail in error.errors():
            cause = detail.get("ctx", {}).get("error")
            if isinstance(cause, ConfigError):
                raise cause from error
        raise


def explain_config_to_yaml(config: ExplainConfig) -> str:
    """Serialize a config to YAML with stable key ordering within sections."""
    payload = config.model_dump(mode="json")
    dumped: str = yaml.safe_dump(payload, sort_keys=False)
    return dumped
