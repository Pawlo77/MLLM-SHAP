"""Optional plugin capabilities detected through protocols.

@see [[knowledge/architecture/pipeline]]
"""

import asyncio
import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import pytest

from mllm_shap import ExplainConfig, ExplainRunner, PluginGroup, PluginRegistry
from mllm_shap.backends.mock import MockBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import (
    ConfigError,
    LogprobsUnavailableError,
    ReanalyzeMismatchError,
)
from mllm_shap.pipeline.manifest import RunManifest, parse_manifest
from mllm_shap.plugins import register_builtin_plugins
from mllm_shap.protocols import (
    FittableValueFunction,
    RequiresEmbeddings,
    SupportsAclose,
    SupportsKvCache,
    SupportsLogprobs,
)
from mllm_shap.protocols.backend import GenerationResult
from mllm_shap.value.embedding import CosineEmbeddingValue, EuclideanEmbeddingValue
from mllm_shap.value.generation_records import generation_record_from_text
from mllm_shap.value.tfidf import TfIdfCosineValue


class CorpusLengthValue:
    """Value function that must be fitted before scoring."""

    instances: ClassVar[list["CorpusLengthValue"]] = []

    def __init__(self, context: object | None = None) -> None:
        self.corpus_sizes: list[int] = []
        CorpusLengthValue.instances.append(self)

    @property
    def name(self) -> str:
        return "corpus_length"

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        self.corpus_sizes.append(len(corpus))

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        assert self.corpus_sizes, "fit must run before score"
        return float(len(candidate.text))


class CachingBackend(MockBackend):
    """Mock backend that also exposes the KV-cache and aclose capabilities."""

    instances: ClassVar[list["CachingBackend"]] = []

    def __init__(self, context: object | None = None) -> None:
        super().__init__(model_id="caching")
        self.kv_settings: list[tuple[bool, int]] = []
        self.closed = False
        self.generations = 0
        CachingBackend.instances.append(self)

    @property
    def kv_cache_hits(self) -> int:
        """Count every generation as a hit, over the backend's lifetime."""
        return self.generations

    @property
    def kv_cache_reused_tokens(self) -> int:
        return 4 * self.generations

    def set_kv_cache_enabled(self, enabled: bool, max_entries: int) -> None:
        self.kv_settings.append((enabled, max_entries))

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        self.generations += 1
        return await super().generate(snapshot, generation)

    async def aclose(self) -> None:
        self.closed = True


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.register(PluginGroup.VALUE_FNS, "corpus_length", CorpusLengthValue)
    registry.register(PluginGroup.BACKENDS, "caching", CachingBackend)
    return registry


def _config(archive_path: str = "") -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "caching", "model_id": "caching"},
        "explanation": {
            "estimator": "exact",
            "value_fn": "corpus_length",
            "absence_policy": "pad",
            "archive": {"path": archive_path},
            "kv_cache": {"enabled": True, "max_entries": 3},
        },
    })


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="a b c"),)),
    ))


def test_capability_protocols_match_builtins() -> None:
    assert isinstance(TfIdfCosineValue(), FittableValueFunction)
    assert not isinstance(MockBackend(), SupportsKvCache)
    assert not isinstance(MockBackend(), SupportsAclose)
    assert isinstance(CachingBackend(), SupportsKvCache)
    assert isinstance(CachingBackend(), SupportsAclose)


def test_pipeline_uses_backend_and_value_capabilities() -> None:
    CachingBackend.instances.clear()
    CorpusLengthValue.instances.clear()
    output = ExplainRunner(_config(), registry=_registry()).explain_sync(_snapshot())

    backend = CachingBackend.instances[-1]
    assert backend.kv_settings == [(True, 3)]
    assert backend.closed
    assert output.metrics.kv_cache_hits == output.metrics.executed
    assert output.metrics.kv_cache_reused_tokens == 4 * output.metrics.executed
    assert CorpusLengthValue.instances[-1].corpus_sizes == [9]


def test_kv_cache_metrics_are_per_run_with_an_injected_backend() -> None:
    backend = CachingBackend()
    with ExplainRunner(_config(), registry=_registry(), backend=backend) as runner:
        first = runner.explain_sync(_snapshot())
        second = runner.explain_sync(_snapshot())

    assert first.metrics.kv_cache_hits > 0
    assert second.metrics.kv_cache_hits == first.metrics.kv_cache_hits
    assert second.metrics.kv_cache_reused_tokens == first.metrics.kv_cache_reused_tokens
    assert first.metrics.kv_cache_hits == first.metrics.executed


def test_reanalyze_fits_custom_fittable_value_function(tmp_path: Path) -> None:
    runner = ExplainRunner(
        _config(archive_path=str(tmp_path / "{run_id}")), registry=_registry()
    )
    first = runner.explain_sync(_snapshot())
    CorpusLengthValue.instances.clear()
    again = runner.reanalyze(tmp_path / first.run_id)

    assert CorpusLengthValue.instances[-1].corpus_sizes == [9]
    assert again.result.values == first.result.values


class EmbeddingBackend(MockBackend):
    """Mock backend that attaches a length-based embedding when asked to."""

    flags: ClassVar[list[bool]] = []

    def __init__(self, context: object | None = None) -> None:
        super().__init__(model_id="embedding")

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        EmbeddingBackend.flags.append(generation.compute_embeddings)
        record = await super().generate(snapshot, generation)
        if not generation.compute_embeddings:
            return record
        vector = (float(len(snapshot.messages()[-1].text)), 1.0)
        return GenerationRecord(
            text=record.text,
            text_token_rows=record.text_token_rows,
            embedding=vector,
            contextual_embedding=vector,
        )


class EmbeddingProbe:
    """Value function that needs embeddings and returns the first component."""

    needs_embeddings = True

    def __init__(self, context: object | None = None) -> None:
        pass

    @property
    def name(self) -> str:
        return "embedding_probe"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        assert isinstance(candidate, GenerationRecord)
        return candidate.embedding[0]


def _embedding_config(value_fn: str) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "embedding", "model_id": "embedding"},
        "explanation": {
            "estimator": "exact",
            "value_fn": value_fn,
            "absence_policy": "delete",
            "archive": {"path": ""},
        },
    })


def test_embedding_value_functions_declare_they_need_embeddings() -> None:
    assert isinstance(CosineEmbeddingValue(), RequiresEmbeddings)
    assert isinstance(EuclideanEmbeddingValue(), RequiresEmbeddings)
    assert CosineEmbeddingValue().needs_embeddings
    assert not isinstance(TfIdfCosineValue(), RequiresEmbeddings)


def test_backend_embeddings_reach_value_functions_that_need_them() -> None:
    registry = _registry()
    registry.register(PluginGroup.BACKENDS, "embedding", EmbeddingBackend)
    registry.register(PluginGroup.VALUE_FNS, "embedding_probe", EmbeddingProbe)
    EmbeddingBackend.flags.clear()

    output = ExplainRunner(
        _embedding_config("embedding_probe"), registry=registry
    ).explain_sync(_snapshot())

    assert EmbeddingBackend.flags
    assert all(EmbeddingBackend.flags)
    # v(S) is the rendered prompt length: "a b c" (5), two players (3), one (1),
    # empty (0); efficiency makes the attributions sum to v(N) - v(empty).
    assert output.result.grand_value == 5.0
    assert output.result.base_value == 0.0
    assert sum(output.result.values) == 5.0


def test_backends_skip_embeddings_for_other_value_functions() -> None:
    registry = _registry()
    registry.register(PluginGroup.BACKENDS, "embedding", EmbeddingBackend)
    EmbeddingBackend.flags.clear()

    ExplainRunner(_embedding_config("tfidf_cosine"), registry=registry).explain_sync(
        _snapshot()
    )

    assert EmbeddingBackend.flags
    assert not any(EmbeddingBackend.flags)


def test_mock_backend_records_match_hash_records() -> None:
    """Without backend embeddings, explain scores the same records as reanalyze."""
    snapshot = _snapshot()
    record = asyncio.run(
        MockBackend().generate(snapshot, GenerationConfig(compute_embeddings=True))
    )
    assert record == generation_record_from_text(record.text)


def test_manifest_records_which_embeddings_were_scored(tmp_path: Path) -> None:
    registry = _registry()
    registry.register(PluginGroup.BACKENDS, "embedding", EmbeddingBackend)
    registry.register(PluginGroup.VALUE_FNS, "embedding_probe", EmbeddingProbe)
    archive = str(tmp_path / "{run_id}")

    def run(kind: str, value_fn: str, mode: str = "static") -> RunManifest:
        payload = _embedding_config(value_fn).model_dump(mode="json")
        payload["backend"]["kind"] = kind
        payload["explanation"]["embedding_mode"] = mode
        payload["explanation"]["archive"]["path"] = archive
        output = ExplainRunner(
            ExplainConfig.model_validate(payload), registry=registry
        ).explain_sync(_snapshot())
        archived = parse_manifest(
            json.loads((tmp_path / output.run_id / "manifest.json").read_text("utf-8"))
        )
        assert archived == output.manifest
        return archived

    backend = run("embedding", "embedding_probe")
    assert (backend.embedding_source, backend.embedding_model) == (
        "backend",
        "embedding",
    )
    # Both mock vectors are the hash embeddings reanalyze scores (C15).
    assert run("mock", "embedding_cosine").embedding_source == "hash"
    contextual = run("mock", "embedding_cosine", "contextual")
    assert (contextual.embedding_source, contextual.embedding_model) == ("hash", "")
    backend_contextual = run("embedding", "embedding_probe", "contextual")
    assert backend_contextual.embedding_source == "backend"
    assert run("embedding", "tfidf_cosine").embedding_source == "none"


@pytest.mark.parametrize("embedding_mode", ["static", "contextual"])
def test_reanalyze_warns_or_refuses_for_backend_embedding_runs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, embedding_mode: str
) -> None:
    registry = _registry()
    registry.register(PluginGroup.BACKENDS, "embedding", EmbeddingBackend)
    payload = _embedding_config("embedding_cosine").model_dump(mode="json")
    payload["explanation"]["embedding_mode"] = embedding_mode
    payload["explanation"]["archive"]["path"] = str(tmp_path / "{run_id}")
    runner = ExplainRunner(ExplainConfig.model_validate(payload), registry=registry)
    first = runner.explain_sync(_snapshot())
    assert first.manifest is not None
    assert first.manifest.embedding_source == "backend"

    with caplog.at_level(logging.WARNING, logger="mllm_shap"):
        runner.reanalyze(tmp_path / first.run_id)
    assert "hash embeddings" in caplog.text
    with pytest.raises(ReanalyzeMismatchError, match="'embedding'"):
        runner.reanalyze(tmp_path / first.run_id, strict=True)


class LogprobBackend(MockBackend):
    """Mock backend whose logprobs encode the prompt: one -1.0 per prompt word."""

    returns_logprobs = True
    drop_logprobs = False

    def __init__(self, context: object | None = None) -> None:
        super().__init__(model_id="logprob")

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        record = await super().generate(snapshot, generation)
        words = len(snapshot.messages()[-1].text.split())
        logprobs = None if self.drop_logprobs else (-2.0 * words, 0.0)
        return dataclasses.replace(record, logprobs=logprobs)


def _logprob_config(kind: str, archive: str = "") -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": kind, "model_id": kind},
        "explanation": {
            "estimator": "exact",
            "value_fn": "logprob",
            "absence_policy": "delete",
            "archive": {"path": archive},
        },
    })


def _logprob_registry() -> PluginRegistry:
    registry = _registry()
    registry.register(PluginGroup.BACKENDS, "logprob", LogprobBackend)
    return registry


def test_logprob_value_function_needs_a_logprob_backend(tmp_path: Path) -> None:
    runner = ExplainRunner(_logprob_config("mock", str(tmp_path / "{run_id}")))
    with pytest.raises(ConfigError, match="'mock'"):
        runner.explain_sync(_snapshot())
    assert not any(tmp_path.iterdir())
    assert not isinstance(MockBackend(), SupportsLogprobs)
    assert isinstance(LogprobBackend(), SupportsLogprobs)


def test_logprob_run_scores_backend_logprobs_and_reanalyze_matches(
    tmp_path: Path,
) -> None:
    runner = ExplainRunner(
        _logprob_config("logprob", str(tmp_path / "{run_id}")),
        registry=_logprob_registry(),
    )
    output = runner.explain_sync(_snapshot())

    # v(S) = mean((-2 * |S|, 0)) = -|S|, so each of the three words gets -1.
    assert output.result.grand_value == -3.0
    assert output.result.base_value == 0.0
    assert output.result.values == pytest.approx((-1.0, -1.0, -1.0))
    again = runner.reanalyze(tmp_path / output.run_id)
    assert again.result.values == pytest.approx(output.result.values)


def test_logprob_run_fails_when_a_record_lacks_logprobs() -> None:
    backend = LogprobBackend()
    backend.drop_logprobs = True
    runner = ExplainRunner(
        _logprob_config("logprob"), registry=_logprob_registry(), backend=backend
    )
    with pytest.raises(LogprobsUnavailableError, match="logprobs"):
        runner.explain_sync(_snapshot())


def test_reanalyze_refuses_logprob_on_archives_without_logprobs(
    tmp_path: Path,
) -> None:
    archive = str(tmp_path / "{run_id}")
    payload = _logprob_config("mock", archive).model_dump(mode="json")
    payload["explanation"]["value_fn"] = "tfidf_cosine"
    output = ExplainRunner(ExplainConfig.model_validate(payload)).explain_sync(
        _snapshot()
    )

    runner = ExplainRunner(_logprob_config("mock", archive))
    with pytest.raises(ReanalyzeMismatchError, match="logprobs"):
        runner.reanalyze(tmp_path / output.run_id)
