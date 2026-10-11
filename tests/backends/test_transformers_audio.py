"""Audio snapshots through the transformers backend, with stub processors.

@see [[src/mllm_shap/backends/transformers/audio.py]]
@see [[src/mllm_shap/backends/transformers/text.py]]
"""

import asyncio
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("soundfile")
pytest.importorskip("librosa")

import torch

from mllm_shap.alignment.io import encode_wav
from mllm_shap.backends.transformers import text as transformers_text
from mllm_shap.backends.transformers.audio import processor_inputs
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context

RATE = 16_000


class _Tokenizer:
    chat_template = ""
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "</s>"

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _Processor:
    """Processor without an audio chat template: audio goes in as ``audio_token``."""

    audio_token = "<|audio|>"
    chat_template: str | None = None

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()
        self.audio_processor = SimpleNamespace(sampling_rate=RATE)
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[Any] = []

    def __call__(self, text: str, audio: Any, return_tensors: str) -> dict[str, Any]:
        self.calls.append({"text": text, "audio": audio})
        return _encoded(len(audio or ()))

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> dict[str, Any]:
        self.conversations.append(conversation)
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        audio = [p for m in conversation for p in m["content"] if p["type"] == "audio"]
        return _encoded(len(audio))


def _encoded(audio_count: int) -> dict[str, Any]:
    ids = torch.tensor([[1, 2, 3, 9 + audio_count]], dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "input_features": torch.ones(1, 2, 3, dtype=torch.float64),
    }


class _Model:
    dtype = torch.float32

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, Any]] = []
        self.forward_calls = 0

    def to(self, device: object) -> "_Model":
        return self

    def eval(self) -> "_Model":
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.forward_calls += 1
        pytest.fail("audio prompts must not run a prefix forward pass")

    def generate(self, **kwargs: Any) -> SimpleNamespace:
        self.generate_calls.append(kwargs)
        suffix = torch.tensor([[7, 8]], device=kwargs["input_ids"].device)
        sequences = torch.cat([kwargs["input_ids"], suffix], dim=1)
        return SimpleNamespace(
            sequences=sequences, logits=tuple(torch.zeros(1, 16) for _ in range(2))
        )


def _speech(seconds: float, rate: int = 8_000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return encode_wav(0.5 * np.sin(2 * np.pi * 220 * t), rate)


def _snapshot(audio: bytes, speaker: str | None = "ann") -> ConversationSnapshot:
    spoken = Message(
        role=Role.USER,
        content=(TextPart("What is said?"), AudioPart(audio, 8_000)),
        speaker=speaker,
    )
    system = Message(role=Role.SYSTEM, text="Be brief.")
    return ConversationSnapshot.from_turns((Turn(messages=(system, spoken)),))


def _backend(processor: _Processor) -> transformers_text.TransformersTextBackend:
    backend = transformers_text.TransformersTextBackend(
        backend_context(BackendConfig(kind="transformers", model_id="org/audio"))
    )
    backend._processor = processor
    backend._tokenizer = processor.tokenizer
    backend._model = _Model()
    backend._device = torch.device("cpu")
    backend.set_kv_cache_enabled(True, 8)
    return backend


def test_audio_token_prompt_puts_one_token_per_audio_part() -> None:
    processor = _Processor()
    processor_inputs(processor, _snapshot(_speech(1.0)))
    (call,) = processor.calls
    assert call["text"] == ("system: Be brief.\nuser: <|audio|>ann: What is said?")
    (waveform,) = call["audio"]
    assert len(waveform) == RATE  # resampled from 8 kHz to the processor's rate


def test_audio_chat_template_gets_audio_and_text_parts() -> None:
    processor = _Processor()
    processor.chat_template = "{% if part.type == 'audio' %}<audio>{% endif %}"
    processor_inputs(processor, _snapshot(_speech(0.5), speaker=None))
    (conversation,) = processor.conversations
    assert conversation[0] == {
        "role": "system",
        "content": [{"type": "text", "text": "Be brief."}],
    }
    audio, text = conversation[1]["content"]
    assert audio["type"] == "audio"
    assert len(audio["audio"]) == RATE // 2
    assert text == {"type": "text", "text": "What is said?"}
    assert processor.calls == []


def test_fully_removed_audio_is_dropped_from_the_prompt() -> None:
    processor = _Processor()
    processor_inputs(processor, _snapshot(encode_wav(np.zeros(0), 8_000)))
    (call,) = processor.calls
    assert "<|audio|>" not in call["text"]
    assert call["audio"] is None


def test_generate_passes_audio_features_and_skips_the_prefix_cache() -> None:
    processor = _Processor()
    backend = _backend(processor)
    record = asyncio.run(
        backend.generate(_snapshot(_speech(0.25)), GenerationConfig(max_new_tokens=2))
    )
    assert record.text == "7 8"
    assert record.logprobs == pytest.approx((math.log(1 / 16),) * 2)
    (kwargs,) = backend._model.generate_calls
    assert kwargs["input_features"].dtype == torch.float32
    assert kwargs["input_ids"].tolist() == [[1, 2, 3, 10]]
    assert backend._model.forward_calls == 0
    assert backend.kv_cache_hits == 0


def test_text_model_refuses_audio_snapshots() -> None:
    backend = _backend(_Processor())
    backend._processor = None
    with pytest.raises(ValueError, match="audio snapshots"):
        asyncio.run(backend.generate(_snapshot(_speech(0.25)), GenerationConfig()))


def test_attn_implementation_reaches_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Granite Speech on MPS needs eager attention (its Q-former breaks under SDPA)."""
    loaded: dict[str, Any] = {}

    class _Auto:
        def __init__(self, result: Any) -> None:
            self._result = result

        def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
            loaded.update(kwargs)
            return self._result

    config = SimpleNamespace(architectures=["GraniteSpeechForConditionalGeneration"])
    fake = SimpleNamespace(
        AutoConfig=_Auto(config),
        MODEL_FOR_CAUSAL_LM_MAPPING={},
        AutoProcessor=_Auto(_Processor()),
        GraniteSpeechForConditionalGeneration=_Auto(_Model()),
    )
    monkeypatch.setattr(transformers_text, "_import_transformers", lambda: fake)
    backend = transformers_text.TransformersTextBackend(
        backend_context(
            BackendConfig(
                kind="transformers",
                model_id="org/audio",
                options={"attn_implementation": "eager"},
            )
        )
    )
    asyncio.run(backend.generate(_snapshot(_speech(0.25)), GenerationConfig()))
    assert loaded["attn_implementation"] == "eager"


@pytest.mark.parametrize(
    ("options", "architecture"),
    [
        ({}, "GraniteSpeechForConditionalGeneration"),
        (
            {"model_class": "OtherForConditionalGeneration"},
            "OtherForConditionalGeneration",
        ),
    ],
)
def test_audio_models_load_a_processor_and_their_architecture(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, str], architecture: str
) -> None:
    calls: list[str] = []

    class _Auto:
        def __init__(self, kind: str, result: Any) -> None:
            self._kind, self._result = kind, result

        def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
            calls.append(self._kind)
            return self._result

    config = SimpleNamespace(architectures=["GraniteSpeechForConditionalGeneration"])
    fake = SimpleNamespace(
        AutoConfig=_Auto("config", config),
        MODEL_FOR_CAUSAL_LM_MAPPING={},
        AutoProcessor=_Auto("processor", _Processor()),
        **{architecture: _Auto(architecture, _Model())},
    )
    monkeypatch.setattr(transformers_text, "_import_transformers", lambda: fake)
    backend = transformers_text.TransformersTextBackend(
        backend_context(
            BackendConfig(kind="transformers", model_id="org/audio", options=options)
        )
    )
    record = asyncio.run(
        backend.generate(_snapshot(_speech(0.25)), GenerationConfig(max_new_tokens=2))
    )
    assert calls == ["config", "processor", architecture]
    assert record.text == "7 8"
