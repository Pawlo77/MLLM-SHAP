"""Tests for LM Studio backend options carried in ``BackendConfig.options``.

@see [[src/mllm_shap/backends/lmstudio/options.py]]
@see [[src/mllm_shap/backends/lmstudio/backend.py]]
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from mllm_shap.backends.lmstudio import LmStudioBackend
from mllm_shap.backends.lmstudio import backend as lmstudio_backend
from mllm_shap.backends.lmstudio.options import LmStudioOptions
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.errors import ConfigError
from mllm_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context


def test_lmstudio_options_defaults() -> None:
    options = LmStudioOptions()
    assert options.hub_search is None
    assert options.quantization == "Q4_K_M"
    assert options.auto_download is True
    assert options.serialize_generate is True


def test_lmstudio_backend_validates_options() -> None:
    backend = LmStudioBackend(
        backend_context(
            BackendConfig(
                kind="lmstudio",
                model_id="m",
                options={"serialize_generate": False, "quantization": "Q8_0"},
            )
        )
    )
    assert backend.options.serialize_generate is False
    assert backend.options.quantization == "Q8_0"


def test_lmstudio_backend_rejects_unknown_option() -> None:
    with pytest.raises(ConfigError, match="hub_serch"):
        LmStudioBackend(
            backend_context(
                BackendConfig(kind="lmstudio", model_id="m", options={"hub_serch": "x"})
            )
        )


def test_backend_config_rejects_lmstudio_fields_at_top_level() -> None:
    with pytest.raises(ValueError, match="serialize_generate"):
        BackendConfig.model_validate({
            "kind": "lmstudio",
            "model_id": "m",
            "serialize_generate": True,
        })


def test_backend_config_has_connection_defaults() -> None:
    config = BackendConfig(kind="api", model_id="m", api_host="http://mock/v1")
    assert config.timeout_s == 120.0
    assert config.max_retries == 3
    assert config.options == {}


def _audio_snapshot() -> ConversationSnapshot:
    spoken = Message(
        role=Role.USER, content=(TextPart("say hi"), AudioPart(b"RIFF", 16_000))
    )
    return ConversationSnapshot.from_turns((Turn(messages=(spoken,)),))


def test_lmstudio_backend_refuses_audio_by_default() -> None:
    backend = LmStudioBackend(
        backend_context(BackendConfig(kind="lmstudio", model_id="m"))
    )
    with pytest.raises(ValueError, match="audio: transcript"):
        asyncio.run(backend.generate(_audio_snapshot(), GenerationConfig()))


def test_lmstudio_transcript_option_sends_the_text_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``audio: transcript`` scores the transcript; the audio is not heard."""
    sent: list[str] = []

    class _Chat:
        def add_user_message(self, text: str) -> None:
            sent.append(text)

    class _Model:
        async def respond(self, chat: _Chat, config: object) -> SimpleNamespace:
            return SimpleNamespace(content="hi")

    backend = LmStudioBackend(
        backend_context(
            BackendConfig(
                kind="lmstudio", model_id="m", options={"audio": "transcript"}
            )
        )
    )

    async def model() -> _Model:
        return _Model()

    monkeypatch.setattr(
        lmstudio_backend, "_import_lmstudio", lambda: SimpleNamespace(Chat=_Chat)
    )
    monkeypatch.setattr(backend, "_ensure_model", model)

    async def run() -> list[str]:
        records = [
            await backend.generate(_audio_snapshot(), GenerationConfig())
            for _ in range(2)
        ]
        return [record.text for record in records]

    with caplog.at_level(logging.WARNING):
        assert asyncio.run(run()) == ["hi", "hi"]
    assert sent == ["say hi", "say hi"]
    warnings = [r for r in caplog.records if "not heard" in r.getMessage()]
    assert len(warnings) == 1
