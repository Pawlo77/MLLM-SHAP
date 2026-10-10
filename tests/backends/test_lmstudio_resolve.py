"""Async tests for LM Studio model resolution.

@see [[src/nlp_shap/backends/lmstudio/__init__.py]]
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nlp_shap.backends.lmstudio import LmStudioBackend
from nlp_shap.backends.lmstudio.models import (
    match_downloaded_model_key,
    resolve_model_key,
)
from nlp_shap.backends.lmstudio.options import LmStudioOptions
from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.errors import BackendUnavailableError
from nlp_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context


def test_resolve_model_key_uses_downloaded_match() -> None:
    """Configured ids resolve to an already-downloaded model key."""
    client = SimpleNamespace(
        llm=SimpleNamespace(
            list_downloaded=AsyncMock(
                return_value=(SimpleNamespace(model_key="qwen2.5-3b-instruct"),),
            ),
            get_model_info=AsyncMock(),
        ),
    )

    async def run() -> str:
        return await resolve_model_key(client, "qwen2.5-3b-instruct", LmStudioOptions())

    key = asyncio.run(run())
    assert key == "qwen2.5-3b-instruct"
    client.llm.get_model_info.assert_not_called()


def test_resolve_model_key_downloads_when_missing() -> None:
    """Missing models are downloaded from the repository when auto_download is on."""
    download_option = SimpleNamespace(
        info=SimpleNamespace(quantization="Q4_K_M", name="q4", recommended=True),
        download=AsyncMock(return_value="qwen2.5-3b-instruct"),
    )
    search_hit = SimpleNamespace(
        search_result=SimpleNamespace(name="Qwen2.5 3B Instruct", exact=True),
        get_download_options=AsyncMock(return_value=(download_option,)),
    )
    client = SimpleNamespace(
        llm=SimpleNamespace(
            list_downloaded=AsyncMock(return_value=()),
            get_model_info=AsyncMock(side_effect=RuntimeError("missing")),
        ),
        repository=SimpleNamespace(
            search_models=AsyncMock(return_value=(search_hit,)),
        ),
    )

    async def run() -> str:
        return await resolve_model_key(client, "qwen2.5-3b-instruct", LmStudioOptions())

    key = asyncio.run(run())
    assert key == "qwen2.5-3b-instruct"
    client.repository.search_models.assert_awaited_once()
    download_option.download.assert_awaited_once()


def _downloaded(*keys: str) -> tuple[SimpleNamespace, ...]:
    return tuple(SimpleNamespace(model_key=key) for key in keys)


def test_match_prefers_exact_over_substring() -> None:
    downloaded = _downloaded("qwen2.5-3b-instruct-mlx", "qwen2.5-3b-instruct")
    assert match_downloaded_model_key(downloaded, "qwen2.5-3b-instruct") == (
        "qwen2.5-3b-instruct"
    )


def test_match_uses_prefix_of_last_path_segment() -> None:
    downloaded = _downloaded("lmstudio-community/qwen2-500m-instruct-gguf", "gemma-2b")
    assert match_downloaded_model_key(downloaded, "qwen2-500m-instruct") == (
        "lmstudio-community/qwen2-500m-instruct-gguf"
    )


def test_match_falls_back_to_substring() -> None:
    downloaded = _downloaded("org/my-qwen2-500m-instruct")
    assert match_downloaded_model_key(downloaded, "qwen2-500m-instruct") == (
        "org/my-qwen2-500m-instruct"
    )


def test_match_does_not_accept_shorter_keys() -> None:
    """A downloaded 'qwen2' no longer matches a request for 'qwen2.5-3b'."""
    assert match_downloaded_model_key(_downloaded("qwen2"), "qwen2.5-3b") is None


def test_match_accepts_a_key_listed_twice() -> None:
    """LM Studio lists one key per downloaded variant; repeats are one model."""
    downloaded = _downloaded("qwen2-500m-instruct", "qwen2-500m-instruct")
    assert match_downloaded_model_key(downloaded, "qwen2-500m-instruct") == (
        "qwen2-500m-instruct"
    )


def test_match_raises_on_ambiguous_candidates() -> None:
    downloaded = _downloaded("a/qwen2-7b-q4", "b/qwen2-7b-q8")
    with pytest.raises(BackendUnavailableError, match=r"a/qwen2-7b-q4.*b/qwen2-7b-q8"):
        match_downloaded_model_key(downloaded, "qwen2-7b")


class _Chat:
    def __init__(self, prompt: str | None = None) -> None:
        self.messages: list[str] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append(text)


class _Model:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.calls = 0

    async def respond(self, chat: object, config: object) -> SimpleNamespace:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return SimpleNamespace(content=" ok ")


class WebsocketClosedError(Exception):
    """Stands in for the SDK's websocket failure."""


def _backend_with_model(
    monkeypatch: pytest.MonkeyPatch, model: _Model
) -> tuple[LmStudioBackend, list[int]]:
    monkeypatch.setattr(
        "nlp_shap.backends.lmstudio.backend._import_lmstudio",
        lambda: SimpleNamespace(Chat=_Chat),
    )
    backend = LmStudioBackend(
        backend_context(BackendConfig(kind="lmstudio", model_id="m"))
    )
    loads: list[int] = []

    async def ensure_model() -> _Model:
        loads.append(1)
        return model

    monkeypatch.setattr(backend, "_ensure_model", ensure_model)
    return backend, loads


def _generate(backend: LmStudioBackend) -> str:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hi"),)),
    ))
    record = asyncio.run(backend.generate(snapshot, GenerationConfig()))
    return record.text


def test_lmstudio_retries_connection_errors_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model([WebsocketClosedError("closed")])
    backend, loads = _backend_with_model(monkeypatch, model)
    assert _generate(backend) == "ok"
    assert model.calls == 2
    assert len(loads) == 2


def test_lmstudio_reraises_value_errors_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model([ValueError("bad prompt")])
    backend, _ = _backend_with_model(monkeypatch, model)
    with pytest.raises(ValueError, match="bad prompt"):
        _generate(backend)
    assert model.calls == 1


def test_lmstudio_does_not_retry_other_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model([RuntimeError("model crashed")])
    backend, _ = _backend_with_model(monkeypatch, model)
    with pytest.raises(BackendUnavailableError, match="generation request failed"):
        _generate(backend)
    assert model.calls == 1
