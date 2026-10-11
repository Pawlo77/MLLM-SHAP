"""Transformers, LM Studio and OpenAI-compatible backends on real models.

LM Studio needs ``MLLM_SHAP_PRACTICAL_LMS_MODEL`` (a downloaded model key such as
``qwen2.5-1.5b-instruct``); the API scenario posts to
``MLLM_SHAP_PRACTICAL_API_HOST`` (default ``http://localhost:1234/v1``, LM Studio's
server) with ``MLLM_SHAP_PRACTICAL_API_MODEL`` (default: the LM Studio key).

@see [[src/mllm_shap/backends/transformers/text.py]]
"""

import asyncio
import json
import os
from typing import Any
from urllib.parse import urlsplit

import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.backends.api import ApiBackend
from mllm_shap.backends.transformers.text import TransformersTextBackend
from mllm_shap.pipeline.config import BackendConfig
from mllm_shap.plugins import PluginContext
from tests.practical._harness import (
    MODELS,
    PracticalModel,
    Scenario,
    scenario_cases,
    summary,
)
from tests.practical._judge import efficiency_gap

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _config(
    scenario: Scenario, backend: BackendConfig, **variant: Any
) -> ExplainConfig:
    config = scenario.config(MODELS[scenario.tier], variant or None)
    return config.model_copy(update={"backend": backend})


@pytest.mark.parametrize("scenario", scenario_cases("backend.transformers"))
def test_transformers_float16_and_auto_dtype_answer_paris(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """The pinned revision loads in float16 and in auto dtype; both say Paris."""
    snapshot = scenario.snapshot
    generation = scenario.config(default_model.spec).generation
    half = asyncio.run(default_model.backend.generate(snapshot, generation))
    spec = default_model.spec
    options = {"revision": spec.revision, "dtype": "auto"}
    auto_config = BackendConfig(
        kind="transformers", model_id=spec.model_id, options=options
    )
    auto = TransformersTextBackend(
        PluginContext(config=ExplainConfig(backend=auto_config))
    )
    try:
        full = asyncio.run(auto.generate(snapshot, generation))
        auto_dtype = str(auto._model.dtype)
    finally:
        asyncio.run(auto.aclose())
    model = default_model.backend._model
    practical_record["properties"] = {
        "float16": half.text,
        "auto": full.text,
        "auto_dtype": auto_dtype,
    }

    assert model.config._commit_hash == spec.revision
    assert str(model.dtype) == "torch.float16"
    for record in (half, full):
        assert scenario.expect["answer"] in record.text
        assert record.logprobs
        assert all(value <= 0.0 for value in record.logprobs)


@pytest.mark.parametrize("scenario", scenario_cases("backend.transformers.kv_cache"))
def test_prefix_cache_leaves_a_three_turn_explanation_unchanged(
    scenario: Scenario, fast_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """In float32 the KV cache on and off give equal values, with hits > 0."""
    config = scenario.config(fast_model.spec)
    exact = TransformersTextBackend(PluginContext(config=config))
    try:
        on = fast_model.explain(scenario, backend=exact)
        off = fast_model.explain(scenario, scenario.variants[0], exact)
    finally:
        asyncio.run(exact.aclose())
    half = [
        fast_model.explain(scenario, variant, fast_model.backend).result.values
        for variant in ({}, scenario.variants[0])
    ]
    practical_record["numbers"] = {"on": summary(on), "off": summary(off)}
    practical_record["properties"] = {
        "kv_cache_hits": on.metrics.kv_cache_hits,
        "float16_max_gap": max(abs(a - b) for a, b in zip(*half, strict=True)),
    }

    assert on.result.values == pytest.approx(off.result.values, abs=1e-12)
    assert on.metrics.kv_cache_hits > 0
    assert off.metrics.kv_cache_hits == 0


def _lms_key() -> str:
    key = os.environ.get("MLLM_SHAP_PRACTICAL_LMS_MODEL")
    if not key:
        pytest.skip(
            "set MLLM_SHAP_PRACTICAL_LMS_MODEL to a downloaded LM Studio model key "
            "(for example qwen2.5-1.5b-instruct)"
        )
    return key


@pytest.mark.parametrize("scenario", scenario_cases("backend.lmstudio"))
def test_lmstudio_serialised_generation_matches_one_inflight(
    scenario: Scenario, practical_record: dict[str, Any]
) -> None:
    """A downloaded key resolves; max_inflight 4 equals max_inflight 1."""
    key = _lms_key()
    lmstudio = pytest.importorskip("lmstudio")
    if asyncio.run(lmstudio.AsyncClient.find_default_local_api_host()) is None:
        pytest.skip("LM Studio API server is not running")
    backend = BackendConfig(
        kind="lmstudio", model_id=key, options={"auto_download": False}
    )
    wide = ExplainRunner(_config(scenario, backend)).explain_sync(scenario.snapshot)
    narrow = ExplainRunner(
        _config(scenario, backend, **scenario.variants[0])
    ).explain_sync(scenario.snapshot)
    practical_record["numbers"] = {
        "inflight_4": summary(wide),
        "inflight_1": summary(narrow),
    }

    assert wide.result.values == pytest.approx(narrow.result.values, abs=1e-12)
    assert wide.metrics.executed == narrow.metrics.executed


def _api_target() -> tuple[str, str]:
    host = os.environ.get("MLLM_SHAP_PRACTICAL_API_HOST", "http://localhost:1234/v1")
    remote = urlsplit(host).hostname not in LOCAL_HOSTS
    if remote and os.environ.get("MLLM_SHAP_PRACTICAL_REMOTE") != "1":
        pytest.skip(f"{host} is remote; set MLLM_SHAP_PRACTICAL_REMOTE=1 to use it")
    model = os.environ.get("MLLM_SHAP_PRACTICAL_API_MODEL") or _lms_key()
    httpx = pytest.importorskip("httpx")
    try:
        httpx.get(f"{host}/models", timeout=3.0).raise_for_status()
    except httpx.HTTPError as error:
        pytest.skip(f"no OpenAI-compatible server at {host}: {error}")
    return host, model


@pytest.mark.parametrize("scenario", scenario_cases("backend.api"))
def test_api_exact_explanation_posts_each_payload_once(
    scenario: Scenario,
    practical_record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact run over /v1: one POST per unique payload, no key, logprobs parsed."""
    host, model = _api_target()
    httpx = pytest.importorskip("httpx")
    monkeypatch.delenv("MLLM_SHAP_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    posted: list[tuple[httpx.Request, dict[str, Any]]] = []

    class Recording(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request: Any) -> Any:
            response = await super().handle_async_request(request)
            await response.aread()
            posted.append((request, json.loads(response.content)))
            return response

    config = _config(scenario, BackendConfig(kind="api", model_id=model, api_host=host))
    backend = ApiBackend(PluginContext(config=config), transport=Recording())

    async def run() -> Any:
        try:
            output = await ExplainRunner(config, backend=backend).explain(
                scenario.snapshot
            )
            record = await backend.generate(scenario.snapshot, config.generation)
            return output, record
        finally:
            await backend.aclose()

    output, record = asyncio.run(run())
    bodies = [request.content for request, _ in posted]
    served = any(body["choices"][0].get("logprobs") for _, body in posted)
    result = output.result
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {"posts": len(bodies), "server_logprobs": served}

    # One POST per coalition plus the base; the grand coalition reuses the base.
    assert len(bodies) == len(set(bodies)) == output.metrics.executed + 1
    assert not any("authorization" in request.headers for request, _ in posted)
    assert efficiency_gap(result.values, result.base_value, result.grand_value) < 1e-9
    assert (record.logprobs is not None) == served
