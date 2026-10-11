"""Contract tests for the OpenAI-compatible API backend.

@see [[src/mllm_shap/backends/api/__init__.py]]
"""

import asyncio
import json

import httpx
import pytest

from mllm_shap.backends.api import ApiBackend
from mllm_shap.backends.api import payload as api_payload
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.errors import BackendUnavailableError
from mllm_shap.pipeline.config import BackendConfig
from mllm_shap.protocols import SupportsLogprobs
from tests.backends._context import backend_context


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def _success_response(text: str) -> httpx.Response:
    body = {
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    return httpx.Response(200, json=body)


def _api_backend(handler, max_retries: int = 3) -> ApiBackend:
    return ApiBackend(
        backend_context(
            BackendConfig(
                kind="api",
                model_id="remote-model",
                api_host="http://mock",
                max_retries=max_retries,
            )
        ),
        transport=httpx.MockTransport(handler),
    )


def test_api_backend_generate_returns_assistant_text() -> None:
    """ApiBackend parses chat-completions JSON into a generation record."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "remote-model"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["temperature"] == 0.0
        return _success_response("world")

    backend = _api_backend(handler)

    async def run() -> str:
        try:
            record = await backend.generate(
                _snapshot("hello"),
                GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1),
            )
            return record.text
        finally:
            await backend.aclose()

    assert asyncio.run(run()) == "world"
    assert calls == 1


def test_api_backend_dedups_identical_requests_at_zero_temperature() -> None:
    """Identical deterministic requests reuse the in-memory response cache."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        del request
        return _success_response("cached")

    backend = _api_backend(handler)
    snapshot = _snapshot("repeat me")

    async def run() -> tuple[str, str]:
        try:
            first = await backend.generate(
                snapshot, GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1)
            )
            second = await backend.generate(
                snapshot, GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1)
            )
            return first.text, second.text
        finally:
            await backend.aclose()

    first_text, second_text = asyncio.run(run())
    assert first_text == second_text == "cached"
    assert calls == 1


def test_api_backend_skips_dedup_when_temperature_is_positive() -> None:
    """Sampling requests always hit the remote endpoint."""
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["temperature"] == 0.7
        return _success_response(f"sample-{calls}")

    backend = _api_backend(handler)
    snapshot = _snapshot("sample me")

    async def run() -> tuple[str, str]:
        try:
            first = await backend.generate(
                snapshot, GenerationConfig(max_new_tokens=4, temperature=0.7, top_k=5)
            )
            second = await backend.generate(
                snapshot, GenerationConfig(max_new_tokens=4, temperature=0.7, top_k=5)
            )
            return first.text, second.text
        finally:
            await backend.aclose()

    first_text, second_text = asyncio.run(run())
    assert first_text == "sample-1"
    assert second_text == "sample-2"
    assert calls == 2


def test_api_backend_raises_on_invalid_response_payload() -> None:
    """Malformed chat-completions payloads raise BackendUnavailableError."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"choices": []})

    backend = _api_backend(handler)

    async def run() -> None:
        try:
            await backend.generate(
                _snapshot("hello"),
                GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
            )
        finally:
            await backend.aclose()

    with pytest.raises(BackendUnavailableError, match="unexpected chat/completions"):
        asyncio.run(run())


def test_api_backend_raises_on_http_error() -> None:
    """HTTP transport failures surface as BackendUnavailableError."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, json={"error": "unavailable"})

    backend = _api_backend(handler)

    async def run() -> None:
        try:
            await backend.generate(
                _snapshot("hello"),
                GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
            )
        finally:
            await backend.aclose()

    with pytest.raises(
        BackendUnavailableError, match="HTTP API generation request failed"
    ):
        asyncio.run(run())


def test_api_backend_rejects_audio_snapshots() -> None:
    """Audio-marked snapshots are rejected before any HTTP call."""
    backend = _api_backend(lambda request: _success_response("noop"))
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(Message(role=Role.USER, content=(AudioPart(b"\x00", 16_000),)),)
        ),
    ))

    async def run() -> None:
        try:
            await backend.generate(
                snapshot, GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1)
            )
        finally:
            await backend.aclose()

    with pytest.raises(ValueError, match="audio snapshots"):
        asyncio.run(run())


def test_payload_cache_key_is_stable_for_equivalent_payloads() -> None:
    """Canonical JSON hashing collapses key-order differences."""
    left = {
        "model": "m",
        "messages": [{"role": "user", "content": "a"}],
        "temperature": 0.0,
    }
    right = {
        "temperature": 0.0,
        "messages": [{"content": "a", "role": "user"}],
        "model": "m",
    }
    assert api_payload.payload_cache_key(left) == api_payload.payload_cache_key(right)


def test_api_backend_raises_when_httpx_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing httpx dependency surfaces BackendUnavailableError."""

    def fail_import() -> None:
        msg = "httpx package is required for the API backend extra"
        raise BackendUnavailableError(msg)

    monkeypatch.setattr("mllm_shap.backends.api.backend._import_httpx", fail_import)
    backend = ApiBackend(
        backend_context(
            BackendConfig(kind="api", model_id="remote-model", api_host="http://mock")
        ),
    )

    async def run() -> None:
        try:
            await backend.generate(
                _snapshot("hello"),
                GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
            )
        finally:
            await backend.aclose()

    with pytest.raises(BackendUnavailableError, match="httpx package is required"):
        asyncio.run(run())


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record retry delays instead of sleeping."""
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("mllm_shap.backends.api.backend.asyncio.sleep", fake_sleep)
    return delays


def _generate(backend: ApiBackend, text: str = "hello") -> str:
    async def run() -> str:
        try:
            record = await backend.generate(_snapshot(text), GenerationConfig())
            return record.text
        finally:
            await backend.aclose()

    return asyncio.run(run())


def test_api_backend_retries_rate_limit_then_succeeds(sleeps: list[float]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _success_response("ok")

    assert _generate(_api_backend(handler)) == "ok"
    assert calls == 2
    assert sleeps == [2.0]


def test_api_backend_caps_retry_after_at_timeout(sleeps: list[float]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "3600"})
        return _success_response("ok")

    backend = ApiBackend(
        backend_context(
            BackendConfig(
                kind="api", model_id="remote-model", api_host="http://mock", timeout_s=5
            )
        ),
        transport=httpx.MockTransport(handler),
    )
    assert _generate(backend) == "ok"
    assert sleeps == [5.0]


def test_api_backend_gives_up_after_max_retries(sleeps: list[float]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    with pytest.raises(BackendUnavailableError, match="request failed"):
        _generate(_api_backend(handler, max_retries=2))
    assert calls == 3
    assert sleeps == [0.5, 1.0]


def test_api_backend_does_not_retry_client_errors(sleeps: list[float]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400)

    with pytest.raises(BackendUnavailableError):
        _generate(_api_backend(handler))
    assert calls == 1
    assert sleeps == []


def test_api_backend_rejects_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(BackendUnavailableError, match="not valid JSON"):
        _generate(_api_backend(handler))


def test_api_backend_response_cache_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("mllm_shap.backends.api.backend.RESPONSE_CACHE_SIZE", 2)
    backend = _api_backend(lambda request: _success_response("x"))

    async def run() -> None:
        try:
            for text in ("a", "b", "c"):
                await backend.generate(_snapshot(text), GenerationConfig())
        finally:
            await backend.aclose()

    asyncio.run(run())
    assert len(backend._response_cache) == 2


@pytest.mark.parametrize(
    ("api_host", "env", "expected"),
    [
        ("http://localhost:1234/v1", {"OPENAI_API_KEY": "sk-openai"}, None),
        ("https://api.openai.com/v1", {"OPENAI_API_KEY": "sk-openai"}, "sk-openai"),
        (
            "https://api.openai.com/v1",
            {"OPENAI_API_KEY": "sk-openai", "MLLM_SHAP_API_KEY": "own"},
            "own",
        ),
        (
            "https://example.test/v1",
            {"OPENAI_API_KEY": "sk-openai", "MLLM_SHAP_API_KEY": "own"},
            "own",
        ),
        ("https://api.openai.com.evil.test/v1", {"OPENAI_API_KEY": "sk"}, None),
    ],
)
def test_api_backend_sends_openai_key_only_to_openai(
    monkeypatch: pytest.MonkeyPatch,
    api_host: str,
    env: dict[str, str],
    expected: str | None,
) -> None:
    """``MLLM_SHAP_API_KEY`` wins; ``OPENAI_API_KEY`` goes to api.openai.com only."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MLLM_SHAP_API_KEY", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization"))
        return _success_response("ok")

    backend = ApiBackend(
        backend_context(BackendConfig(kind="api", model_id="m", api_host=api_host)),
        transport=httpx.MockTransport(handler),
    )

    assert _generate(backend) == "ok"
    assert seen == [None if expected is None else f"Bearer {expected}"]


def _generate_twice(handler) -> list[GenerationRecord]:
    backend = _api_backend(handler)

    async def run() -> list[GenerationRecord]:
        try:
            generation = GenerationConfig(max_new_tokens=8)
            return [
                await backend.generate(_snapshot("hello"), generation) for _ in range(2)
            ]
        finally:
            await backend.aclose()

    return asyncio.run(run())


def test_api_backend_requests_and_parses_logprobs() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content.decode("utf-8"))["logprobs"] is True
        choice = {
            "message": {"role": "assistant", "content": "blue sky"},
            "logprobs": {
                "content": [
                    {"token": "blue", "logprob": -0.25},
                    {"token": " sky", "logprob": -1.5},
                ]
            },
        }
        return httpx.Response(200, json={"choices": [choice]})

    first, cached = _generate_twice(handler)
    assert first.logprobs == (-0.25, -1.5)
    assert cached.logprobs == (-0.25, -1.5)
    assert isinstance(_api_backend(handler), SupportsLogprobs)


def test_api_backend_scores_an_empty_completion_with_no_logprobs() -> None:
    """A zero-token reply (``logprobs: null`` from LM Studio) has logprobs ``()``."""

    def handler(request: httpx.Request) -> httpx.Response:
        choice = {"message": {"role": "assistant", "content": ""}, "logprobs": None}
        return httpx.Response(200, json={"choices": [choice]})

    first, _ = _generate_twice(handler)
    assert first.text == ""
    assert first.logprobs == ()


def test_api_backend_returns_none_when_the_server_omits_logprobs() -> None:
    first, _ = _generate_twice(lambda request: _success_response("world"))
    assert first.text == "world"
    assert first.logprobs is None
