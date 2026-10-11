"""Versioned run manifest persisted next to every run archive.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_manifest.py]]
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import TypedDict

from ..domain.estimands import Estimand, EstimandWire, estimand_to_wire
from ..errors import ArchiveCorruptError
from ..runtime.archive import MANIFEST_SCHEMA_VERSION, check_manifest_schema_version
from .config import ExplainConfig

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "EmbeddingSource",
    "RunManifest",
    "RunManifestPayload",
    "parse_manifest",
]


class EmbeddingSource(StrEnum):
    """Which embeddings the value function of a run scored."""

    NONE = "none"
    """The value function read no embeddings."""

    HASH = "hash"
    """Hash embeddings derived from the generated text, as ``reanalyze`` scores."""

    BACKEND = "backend"
    """Model embeddings computed by the backend; ``reanalyze`` cannot reproduce them."""


def _package_version() -> str:
    try:
        return version("mllm-shap")
    except PackageNotFoundError:
        return "0.0.0+unknown"


class RunManifestPayload(TypedDict):
    """JSON-compatible manifest payload persisted with a run archive."""

    schema_version: int
    """Manifest schema version; only ``MANIFEST_SCHEMA_VERSION`` is accepted."""

    mllm_shap_version: str
    """Package version that wrote the manifest."""

    run_id: str
    """Stable identifier for the explain run."""

    snapshot_id: str
    """Identifier of the explained conversation snapshot."""

    estimand: EstimandWire
    """Estimand label recorded for every attribution output."""

    estimator: str
    """Registered estimator plugin that sampled the coalitions."""

    value_fn: str
    """Registered value function that scored the coalitions."""

    normalizer: str
    """Registered normalizer applied after aggregation."""

    absence_policy: str
    """Registered absence policy used to render masked snapshots."""

    players: str
    """Registered partition plugin that defined the players."""

    backend_kind: str
    """Registered backend kind that generated the coalition outputs."""

    model_id: str
    """Backend model identifier that generated the coalition outputs."""

    player_ids: list[str]
    """Player identifiers in mask bit order."""

    embedding_source: str
    """Which embeddings the run scored, as an :class:`EmbeddingSource` value."""

    embedding_model: str
    """Model that computed backend embeddings; empty unless ``"backend"``."""

    config: dict[str, object]
    """Full explain configuration echo (``ExplainConfig.model_dump(mode="json")``)."""


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Top-level metadata persisted alongside a run archive.

    Only ``estimand`` and ``run_id`` are required so hand-built archives stay
    simple; the explain pipeline fills every field.
    """

    estimand: Estimand
    """Estimand label recorded for every attribution output."""

    run_id: str
    """Stable identifier for the explain run."""

    snapshot_id: str = ""
    """Identifier of the explained conversation snapshot."""

    estimator: str = ""
    """Registered estimator plugin that sampled the coalitions."""

    value_fn: str = ""
    """Registered value function that scored the coalitions."""

    normalizer: str = ""
    """Registered normalizer applied after aggregation."""

    absence_policy: str = ""
    """Registered absence policy used to render masked snapshots."""

    players: str = ""
    """Registered partition plugin that defined the players."""

    backend_kind: str = ""
    """Registered backend kind that generated the coalition outputs."""

    model_id: str = ""
    """Backend model identifier that generated the coalition outputs."""

    player_ids: tuple[str, ...] = ()
    """Player identifiers in mask bit order."""

    embedding_source: EmbeddingSource = EmbeddingSource.NONE
    """Which embeddings the value function scored."""

    embedding_model: str = ""
    """Backend model that computed the embeddings when ``embedding_source`` is
    ``"backend"``; empty otherwise."""

    config: ExplainConfig | None = None
    """Explain configuration of the run, when the pipeline wrote the manifest."""

    mllm_shap_version: str = field(default_factory=_package_version)
    """Package version that wrote the manifest."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    """Manifest schema version."""

    def to_dict(self) -> RunManifestPayload:
        """Serialize manifest fields to JSON-compatible primitives."""
        return {
            "schema_version": self.schema_version,
            "mllm_shap_version": self.mllm_shap_version,
            "run_id": self.run_id,
            "snapshot_id": self.snapshot_id,
            "estimand": estimand_to_wire(self.estimand),
            "estimator": self.estimator,
            "value_fn": self.value_fn,
            "normalizer": self.normalizer,
            "absence_policy": self.absence_policy,
            "players": self.players,
            "backend_kind": self.backend_kind,
            "model_id": self.model_id,
            "player_ids": list(self.player_ids),
            "embedding_source": self.embedding_source.value,
            "embedding_model": self.embedding_model,
            "config": {}
            if self.config is None
            else self.config.model_dump(mode="json"),
        }


def parse_manifest(payload: Mapping[str, object]) -> RunManifest:
    """Parse a JSON manifest payload into a run manifest.

    A missing ``embedding_source`` reads as ``"none"`` and a missing
    ``embedding_model`` as empty.

    Raises:
        ManifestVersionError: If ``schema_version`` is missing or differs from
            ``MANIFEST_SCHEMA_VERSION``.
        ArchiveCorruptError: If a field has the wrong type or value, the
            recorded config is invalid, or the manifest records a config but
            lacks the run fields the explain pipeline always writes.
    """
    check_manifest_schema_version(payload)
    player_ids = payload.get("player_ids", [])
    config_payload = payload.get("config", {})
    if not isinstance(player_ids, list) or not isinstance(config_payload, dict):
        msg = "manifest player_ids must be a list and config a mapping"
        raise ArchiveCorruptError(msg)
    if config_payload:
        missing = [
            key for key in ("snapshot_id", "estimator") if not _text(payload, key)
        ]
        if missing:
            msg = (
                f"manifest has a config but empty {', '.join(missing)}; "
                "it was not written by ExplainRunner"
            )
            raise ArchiveCorruptError(msg)
    try:
        embedding_source = EmbeddingSource(
            _text(payload, "embedding_source") or EmbeddingSource.NONE
        )
        estimand = Estimand(_text(payload, "estimand"))
        config = (
            ExplainConfig.model_validate(config_payload) if config_payload else None
        )
    except ValueError as exc:
        msg = f"manifest embedding_source, estimand or config is invalid: {exc}"
        raise ArchiveCorruptError(msg) from exc
    return RunManifest(
        estimand=estimand,
        run_id=_text(payload, "run_id"),
        snapshot_id=_text(payload, "snapshot_id"),
        estimator=_text(payload, "estimator"),
        value_fn=_text(payload, "value_fn"),
        normalizer=_text(payload, "normalizer"),
        absence_policy=_text(payload, "absence_policy"),
        players=_text(payload, "players"),
        backend_kind=_text(payload, "backend_kind"),
        model_id=_text(payload, "model_id"),
        player_ids=tuple(str(player_id) for player_id in player_ids),
        embedding_source=embedding_source,
        embedding_model=_text(payload, "embedding_model"),
        config=config,
        mllm_shap_version=_text(payload, "mllm_shap_version"),
        schema_version=MANIFEST_SCHEMA_VERSION,
    )


def _text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        msg = f"manifest field {key!r} must be a string"
        raise ArchiveCorruptError(msg)
    return value
