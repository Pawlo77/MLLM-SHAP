"""Tests for the versioned run manifest schema.

@see [[src/mllm_shap/pipeline/manifest.py]]
"""

import dataclasses

import pytest

from mllm_shap import ExplainConfig
from mllm_shap.domain.estimands import Estimand
from mllm_shap.errors import ArchiveCorruptError, ManifestVersionError
from mllm_shap.pipeline.manifest import (
    MANIFEST_SCHEMA_VERSION,
    EmbeddingSource,
    RunManifest,
    parse_manifest,
)


def _config() -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "value_fn": "tfidf_cosine",
            "absence_policy": "pad",
            "archive": {"path": ""},
        },
    })


def _manifest() -> RunManifest:
    config = _config()
    explanation = config.explanation
    return RunManifest(
        estimand=explanation.estimand,
        run_id="run-7",
        snapshot_id="snap-1",
        estimator=explanation.estimator,
        value_fn=explanation.value_fn,
        normalizer=explanation.normalizer,
        absence_policy=explanation.absence_policy,
        players=explanation.players,
        backend_kind=config.backend.kind,
        model_id=config.backend.model_id,
        player_ids=("snap-1:0:0:0", "snap-1:0:0:1"),
        config=config,
    )


def test_run_manifest_requires_estimand() -> None:
    """Manifest dataclass always includes estimand."""
    manifest = RunManifest(estimand=Estimand.BANZHAF, run_id="run-1")
    assert manifest.estimand is Estimand.BANZHAF
    assert manifest.run_id == "run-1"
    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION


def test_run_manifest_round_trips_every_field() -> None:
    """Serialized manifests parse back to an equal manifest."""
    manifest = _manifest()
    payload = manifest.to_dict()
    assert payload["schema_version"] == 5
    assert payload["estimand"] == "shapley"
    assert payload["estimator"] == "exact"
    assert payload["player_ids"] == ["snap-1:0:0:0", "snap-1:0:0:1"]
    assert parse_manifest(payload) == manifest


def test_parse_manifest_rejects_v1() -> None:
    """Version-1 manifests (estimand and run id only) are refused."""
    with pytest.raises(ManifestVersionError):
        parse_manifest({"estimand": "shapley", "run_id": "x"})


def test_parse_manifest_requires_run_fields_when_config_present() -> None:
    """A manifest with a config but no estimator was not written by the runner."""
    payload = dict(_manifest().to_dict())
    payload["estimator"] = ""
    with pytest.raises(ValueError, match="not written by ExplainRunner"):
        parse_manifest(payload)


def test_parse_manifest_rejects_v2() -> None:
    payload = dict(_manifest().to_dict())
    payload["schema_version"] = 2
    with pytest.raises(ManifestVersionError, match="Re-run explain"):
        parse_manifest(payload)


def test_manifest_round_trips_embedding_source() -> None:
    manifest = dataclasses.replace(
        _manifest(),
        embedding_source=EmbeddingSource.BACKEND,
        embedding_model="org/model",
    )
    payload = manifest.to_dict()
    assert payload["embedding_source"] == "backend"
    assert payload["embedding_model"] == "org/model"
    assert parse_manifest(payload) == manifest


def test_parse_manifest_defaults_and_validates_embedding_source() -> None:
    payload = dict(_manifest().to_dict())
    del payload["embedding_source"]
    del payload["embedding_model"]
    assert parse_manifest(payload).embedding_source == "none"
    payload["embedding_source"] = "word2vec"
    with pytest.raises(ArchiveCorruptError, match="embedding_source"):
        parse_manifest(payload)
