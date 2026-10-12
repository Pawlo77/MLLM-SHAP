"""Audio snapshots through the transformers backend, with stub processors.

Only the generation test needs torch; the rest runs with the ``audio`` extra.

@see [[src/mllm_shap/backends/transformers/audio.py]]
@see [[src/mllm_shap/backends/transformers/text.py]]
"""

import asyncio
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("soundfile")
pytest.importorskip("librosa")

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
from mllm_shap.errors import BackendUnavailableError
from mllm_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context

RATE = 16_000


class _Tensor:
    """Records the device and dtype it is moved to."""

    def __init__(self, floating: bool) -> None:
        self.floating = floating
        self.moved: tuple[Any, Any] | None = None

    def is_floating_point(self) -> bool:
        return self.floating

    def to(self, device: Any, dtype: Any = None) -> "_Tensor":
        self.moved = (device, dtype)
        return self


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

    def __init__(self, encode: Any = None) -> None:
        self.tokenizer = _Tokenizer()
        self.audio_processor = SimpleNamespace(sampling_rate=RATE)
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[Any] = []
        self._encode = encode or (lambda count: {"count": count})

    def __call__(self, text: str, audio: Any, return_tensors: str) -> Any:
        self.calls.append({"text": text, "audio": audio})
        return self._encode(len(audio or ()))

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        self.conversations.append(conversation)
        assert kwargs["tokenize"] is True
        assert kwargs["return_dict"] is True
        audio = [p for m in conversation for p in m["content"] if p["type"] == "audio"]
        return self._encode(len(audio))


class _Model:
    dtype = "model-dtype"

    def to(self, device: object) -> "_Model":
        return self

    def eval(self) -> "_Model":
        return self


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


def _backend(
    processor: _Processor, model: Any
) -> transformers_text.TransformersTextBackend:
    backend = transformers_text.TransformersTextBackend(
        backend_context(BackendConfig(kind="transformers", model_id="org/audio"))
    )
    backend._processor = processor
    backend._tokenizer = processor.tokenizer
    backend._model = model
    backend._device = "device"
    backend.set_kv_cache_enabled(True, 8)
    return backend


def test_audio_token_prompt_puts_one_token_per_audio_part() -> None:
    processor = _Processor()
    processor_inputs(processor, _snapshot(_speech(1.0)))
    (call,) = processor.calls
    assert call["text"] == "system: Be brief.\nuser: <|audio|>ann: What is said?"
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


def test_feature_extractor_rate_is_read_too() -> None:
    """Processors that name their audio component ``feature_extractor`` work."""
    processor = _Processor()
    del processor.audio_processor
    processor.feature_extractor = SimpleNamespace(sampling_rate=8_000)
    processor_inputs(processor, _snapshot(_speech(1.0)))
    assert len(processor.calls[0]["audio"][0]) == 8_000


def test_fully_removed_audio_is_dropped_from_the_prompt() -> None:
    processor = _Processor()
    processor_inputs(processor, _snapshot(encode_wav(np.zeros(0), 8_000)))
    (call,) = processor.calls
    assert "<|audio|>" not in call["text"]
    assert call["audio"] is None


def test_encode_moves_features_to_the_model_dtype_and_device() -> None:
    """Float features take the model's dtype; integer ids only move device."""
    ids, features = _Tensor(floating=False), _Tensor(floating=True)
    processor = _Processor(lambda count: {"input_ids": ids, "input_features": features})
    backend = _backend(processor, _Model())
    snapshot = _snapshot(_speech(0.25))
    encoded = backend._encode(None, _Model(), processor.tokenizer, snapshot)
    assert set(encoded) == {"input_ids", "input_features"}
    assert ids.moved == ("device", None)
    assert features.moved == ("device", "model-dtype")


def test_text_model_refuses_audio_snapshots() -> None:
    backend = _backend(_Processor(), _Model())
    backend._processor = None
    with pytest.raises(ValueError, match="audio snapshots"):
        asyncio.run(backend.generate(_snapshot(_speech(0.25)), GenerationConfig()))


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        float32="float32",
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )


class _CausalConfig:
    architectures = ("GPT2LMHeadModel",)


@pytest.mark.parametrize(
    ("config", "options", "loaded"),
    [
        (
            SimpleNamespace(architectures=["GraniteSpeechForConditionalGeneration"]),
            {"attn_implementation": "eager"},
            ["config", "processor", "GraniteSpeechForConditionalGeneration"],
        ),
        (
            _CausalConfig(),
            {"model_class": "Gemma3nForConditionalGeneration"},
            ["config", "processor", "Gemma3nForConditionalGeneration"],
        ),
        (_CausalConfig(), {}, ["config", "tokenizer", "causal"]),
    ],
)
def test_loading_picks_the_processor_path_for_unknown_causal_models(
    monkeypatch: pytest.MonkeyPatch,
    config: Any,
    options: dict[str, str],
    loaded: list[str],
) -> None:
    """``AutoModelForCausalLM`` checkpoints load as text unless ``model_class``."""
    calls: list[str] = []
    kwargs_seen: dict[str, Any] = {}

    class _Auto:
        def __init__(self, kind: str, result: Any) -> None:
            self._kind, self._result = kind, result

        def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
            calls.append(self._kind)
            kwargs_seen.update(kwargs)
            return self._result

    processor = _Processor()
    fake = SimpleNamespace(
        AutoConfig=_Auto("config", config),
        MODEL_FOR_CAUSAL_LM_MAPPING={_CausalConfig: object},
        AutoProcessor=_Auto("processor", processor),
        AutoTokenizer=_Auto("tokenizer", _Tokenizer()),
        AutoModelForCausalLM=_Auto("causal", _Model()),
        GraniteSpeechForConditionalGeneration=_Auto(
            "GraniteSpeechForConditionalGeneration", _Model()
        ),
        Gemma3nForConditionalGeneration=_Auto(
            "Gemma3nForConditionalGeneration", _Model()
        ),
    )
    monkeypatch.setattr(transformers_text, "_import_transformers", lambda: fake)
    monkeypatch.setattr(transformers_text, "import_torch", _fake_torch)
    backend = transformers_text.TransformersTextBackend(
        backend_context(
            BackendConfig(kind="transformers", model_id="org/m", options=options)
        )
    )
    asyncio.run(backend._ensure_loaded())
    assert calls == loaded
    assert kwargs_seen.get("attn_implementation") == options.get("attn_implementation")
    assert (backend._processor is processor) == ("processor" in loaded)


def test_generate_passes_audio_features_and_skips_the_prefix_cache() -> None:
    torch = pytest.importorskip("torch")

    def encode(count: int) -> dict[str, Any]:
        ids = torch.tensor([[1, 2, 3, 9 + count]], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "input_features": torch.ones(1, 2, 3, dtype=torch.float64),
        }

    class _TorchModel:
        dtype = torch.float32

        def __init__(self) -> None:
            self.generate_calls: list[dict[str, Any]] = []

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("audio prompts must not run a prefix forward pass")

        def generate(self, **kwargs: Any) -> SimpleNamespace:
            self.generate_calls.append(kwargs)
            suffix = torch.tensor([[7, 8]], device=kwargs["input_ids"].device)
            sequences = torch.cat([kwargs["input_ids"], suffix], dim=1)
            logits = tuple(torch.zeros(1, 16) for _ in range(2))
            return SimpleNamespace(sequences=sequences, logits=logits)

    model = _TorchModel()
    backend = _backend(_Processor(encode), model)
    backend._device = torch.device("cpu")
    record = asyncio.run(
        backend.generate(_snapshot(_speech(0.25)), GenerationConfig(max_new_tokens=2))
    )
    assert record.text == "7 8"
    assert record.logprobs == pytest.approx((math.log(1 / 16),) * 2)
    (kwargs,) = model.generate_calls
    assert kwargs["input_features"].dtype == torch.float32
    assert kwargs["input_ids"].tolist() == [[1, 2, 3, 10]]
    assert backend.kv_cache_hits == 0


def test_audio_only_message_sends_no_text_part() -> None:
    processor = _Processor()
    processor.chat_template = "audio"
    spoken = Message(role=Role.USER, content=(AudioPart(_speech(0.25), 8_000),))
    processor_inputs(
        processor, ConversationSnapshot.from_turns((Turn(messages=(spoken,)),))
    )
    ((message,),) = processor.conversations
    assert [part["type"] for part in message["content"]] == ["audio"]


class _TextTokenizer(_Tokenizer):
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        return self.ids


def test_encode_tokenizes_text_prompts_without_the_processor() -> None:
    """Text snapshots give ``input_ids`` alone, so the prefix cache can apply."""
    torch = SimpleNamespace(
        long="long", tensor=lambda data, dtype, device: (data, dtype, device)
    )
    backend = _backend(_Processor(), _Model())
    text = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="hi"),)),
    ))
    encoded = backend._encode(torch, _Model(), _TextTokenizer([4, 5]), text)
    assert encoded == {"input_ids": ([[4, 5]], "long", "device")}
    with pytest.raises(BackendUnavailableError, match="empty prompt"):
        backend._encode(torch, _Model(), _TextTokenizer([]), text)


def test_loading_failure_raises_backend_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Missing:
        def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
            raise OSError(model_id)

    fake = SimpleNamespace(AutoConfig=_Missing())
    monkeypatch.setattr(transformers_text, "_import_transformers", lambda: fake)
    monkeypatch.setattr(transformers_text, "import_torch", _fake_torch)
    backend = transformers_text.TransformersTextBackend(
        backend_context(BackendConfig(kind="transformers", model_id="org/missing"))
    )
    with pytest.raises(BackendUnavailableError, match="org/missing"):
        asyncio.run(backend._ensure_loaded())
