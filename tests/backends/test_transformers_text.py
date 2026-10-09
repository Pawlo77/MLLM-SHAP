"""Tests for the Hugging Face transformers text backend.

@see [[src/nlp_shap/backends/transformers/text.py]]
"""

import asyncio
import dataclasses
import math
from types import SimpleNamespace
from typing import Any

import pytest

from nlp_shap.backends.transformers import text as transformers_text
from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    Turn,
)
from nlp_shap.domain.enums import Role
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.errors import BackendUnavailableError, ConfigError
from nlp_shap.pipeline.config import BackendConfig
from nlp_shap.protocols import SupportsLogprobs
from nlp_shap.value.generation_records import generation_record_from_text
from tests.backends._context import backend_context

PROMPT_IDS = {
    "alpha": [1, 2, 3, 4, 5, 6],
    "beta": [1, 2, 3, 4, 5, 7],
    "gamma": [1, 2, 3, 8, 9, 10],
    "delta": [1, 2, 3, 8, 9, 11],
}
"""Token ids the stub tokenizer returns, keyed by the prompt's last word."""


class _TokenizerStub:
    pad_token_id = 0
    eos_token_id = 0
    eos_token = "</s>"
    pad_token = "</s>"

    def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return list(PROMPT_IDS[prompt.split()[-1]])

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _CacheStub:
    """Croppable KV state that records the token ids it holds."""

    def __init__(self) -> None:
        self.tokens: list[int] = []

    def crop(self, max_length: int) -> None:
        self.tokens = self.tokens[:max_length]


class _ModelStub:
    def __init__(self) -> None:
        self.dtype: Any = "float32"
        self.generate_calls: list[dict[str, Any]] = []
        self.forward_calls: list[dict[str, Any]] = []
        self.generate_past_tokens: list[list[int] | None] = []

    def to(self, device: object) -> "_ModelStub":
        del device
        return self

    def eval(self) -> "_ModelStub":
        return self

    def parameters(self) -> Any:
        torch = pytest.importorskip("torch")
        yield torch.zeros(1)

    def get_input_embeddings(self) -> Any:
        torch = pytest.importorskip("torch")

        def embed(ids: Any) -> Any:
            return ids.to(dtype=torch.float32).unsqueeze(-1)

        return embed

    def __call__(self, input_ids: Any, **kwargs: Any) -> SimpleNamespace:
        torch = pytest.importorskip("torch")
        self.forward_calls.append({"input_ids": input_ids, **kwargs})
        hidden = input_ids.to(dtype=torch.float32).unsqueeze(-1)
        cache = None
        if kwargs.get("use_cache"):
            cache = kwargs.get("past_key_values") or _CacheStub()
            cache.tokens.extend(input_ids[0].tolist())
        return SimpleNamespace(past_key_values=cache, hidden_states=(hidden,))

    def generate(self, **kwargs: Any) -> Any:
        torch = pytest.importorskip("torch")
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        prompt = input_ids[0].tolist()
        past = kwargs.get("past_key_values")
        suffix = [7, 8]
        if past is None:
            self.generate_past_tokens.append(None)
        else:
            # Known truth: a reused state holds a proper prefix of the prompt.
            assert past.tokens == prompt[: len(past.tokens)]
            assert len(past.tokens) < len(prompt)
            self.generate_past_tokens.append(list(past.tokens))
            # Like Hugging Face, generation appends to the state it was given.
            past.tokens.extend(prompt[len(past.tokens) :] + suffix)
        suffix_ids = torch.tensor([suffix], dtype=torch.long, device=input_ids.device)
        sequences = torch.cat([input_ids, suffix_ids], dim=1)
        # Uniform logits over a 16-token vocabulary: every token has log(1/16).
        logits = tuple(torch.zeros(1, 16) for _ in suffix)
        return SimpleNamespace(sequences=sequences, logits=logits)


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def _backend_with_stub(
    kv_cache: bool = True,
) -> transformers_text.TransformersTextBackend:
    torch = pytest.importorskip("torch")
    backend = transformers_text.TransformersTextBackend(
        backend_context(BackendConfig(kind="transformers", model_id="stub-model")),
    )
    backend._tokenizer = _TokenizerStub()
    backend._model = _ModelStub()
    backend._device = torch.device("cpu")
    backend.set_kv_cache_enabled(kv_cache, 8)
    return backend


def test_transformers_backend_generate_is_deterministic_at_zero_temperature() -> None:
    """Identical snapshots produce identical generations when temperature is zero."""
    backend = _backend_with_stub()
    snapshot = _snapshot("shared alpha")

    async def run() -> tuple[str, str]:
        first = await backend.generate(
            snapshot, GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1)
        )
        second = await backend.generate(
            snapshot, GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1)
        )
        return first.text, second.text

    first_text, second_text = asyncio.run(run())
    assert first_text == second_text


def test_transformers_backend_varies_with_masked_text() -> None:
    """Different masked snapshots produce different token prompts."""
    backend = _backend_with_stub()

    async def run() -> None:
        await backend.generate(
            _snapshot("shared alpha"),
            GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
        )
        await backend.generate(
            _snapshot("shared beta"),
            GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
        )

    asyncio.run(run())
    first_ids = backend._model.generate_calls[0]["input_ids"].tolist()
    second_ids = backend._model.generate_calls[1]["input_ids"].tolist()
    assert first_ids != second_ids


def test_transformers_backend_rejects_audio_snapshots() -> None:
    """Audio-marked snapshots raise before model generation."""
    backend = _backend_with_stub()
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(Message(role=Role.USER, content=(AudioPart(b"\x00", 16_000),)),)
        ),
    ))

    with pytest.raises(ValueError, match="audio snapshots"):
        asyncio.run(
            backend.generate(
                snapshot, GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1)
            )
        )


def test_transformers_backend_populates_embedding_fields_on_request() -> None:
    """``compute_embeddings`` makes records carry model embedding vectors."""
    backend = _backend_with_stub()

    async def run() -> GenerationRecord:
        return await backend.generate(
            _snapshot("shared alpha"),
            GenerationConfig(max_new_tokens=4, compute_embeddings=True),
        )

    record = asyncio.run(run())
    assert record.embedding == (7.5,)
    assert record.contextual_embedding == (7.5,)


def test_transformers_backend_skips_embeddings_by_default() -> None:
    """Without ``compute_embeddings`` no embedding forward pass runs."""
    backend = _backend_with_stub(kv_cache=False)

    async def run() -> GenerationRecord:
        return await backend.generate(
            _snapshot("shared alpha"), GenerationConfig(max_new_tokens=4)
        )

    record = asyncio.run(run())
    assert record == dataclasses.replace(
        generation_record_from_text("7 8"), logprobs=record.logprobs
    )
    assert not [
        call for call in backend._model.forward_calls if "output_hidden_states" in call
    ]


def test_transformers_backend_reports_kv_cache_hits() -> None:
    """Shared token prefixes increment kv_cache_hits on subsequent generations."""
    backend = _backend_with_stub()

    async def run() -> int:
        await backend.generate(
            _snapshot("shared alpha"),
            GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
        )
        await backend.generate(
            _snapshot("shared beta"),
            GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
        )
        return backend.kv_cache_hits

    hits = asyncio.run(run())
    assert hits > 0


@pytest.mark.parametrize(
    ("dtype", "warnings"), [("float16", 1), ("bfloat16", 1), ("float32", 0)]
)
def test_transformers_backend_warns_once_on_half_precision_prefix_cache(
    caplog: pytest.LogCaptureFixture, dtype: str, warnings: int
) -> None:
    """D04: a half-precision prefix cache rounds differently, so it warns once."""
    torch = pytest.importorskip("torch")
    backend = _backend_with_stub()
    backend._model.dtype = getattr(torch, dtype)
    generation = GenerationConfig(max_new_tokens=4)

    async def run() -> None:
        for word in ("alpha", "beta"):
            await backend.generate(_snapshot(f"shared {word}"), generation)

    with caplog.at_level("WARNING", logger=transformers_text.__name__):
        asyncio.run(run())
    messages = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(messages) == warnings
    assert all("float32" in message for message in messages)


def test_transformers_backend_raises_when_optional_extra_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing transformers dependencies surface BackendUnavailableError."""

    def fail_import() -> Any:
        msg = "blocked"
        raise ImportError(msg)

    monkeypatch.setattr(transformers_text, "_import_transformers", fail_import)
    monkeypatch.setattr(transformers_text, "import_torch", fail_import)
    backend = transformers_text.TransformersTextBackend(
        backend_context(BackendConfig(kind="transformers", model_id="stub-model")),
    )

    with pytest.raises(BackendUnavailableError, match="transformers backend requires"):
        asyncio.run(
            backend.generate(
                _snapshot("hello"),
                GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
            )
        )


def test_transformers_backend_reuses_prefix_kv_state() -> None:
    """Misses generate from their fresh prefix state; partial hits extend it."""
    backend = _backend_with_stub()
    generation = GenerationConfig(max_new_tokens=4)

    async def run() -> None:
        for word in ("alpha", "beta", "gamma", "delta"):
            await backend.generate(_snapshot(f"shared {word}"), generation)

    asyncio.run(run())
    model = backend._model
    prefix_forwards = [
        call["input_ids"].tolist()
        for call in model.forward_calls
        if call.get("use_cache")
    ]
    # alpha: miss, one forward over its 5-token prefix. gamma: shares 3 tokens
    # with alpha's entry, so only its 2 remaining prefix tokens are computed.
    assert prefix_forwards == [[[1, 2, 3, 4, 5]], [[8, 9]]]
    assert model.generate_past_tokens == [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 8, 9],
        [1, 2, 3, 8, 9],
    ]
    assert backend.kv_cache_hits == 3
    assert backend.kv_cache_reused_tokens == 5 + 3 + 5


def test_transformers_backend_without_kv_cache_runs_no_prefix_forward() -> None:
    backend = _backend_with_stub(kv_cache=False)
    generation = GenerationConfig(max_new_tokens=4)

    async def run() -> None:
        await backend.generate(_snapshot("shared alpha"), generation)
        await backend.generate(_snapshot("shared beta"), generation)

    asyncio.run(run())
    model = backend._model
    assert not [call for call in model.forward_calls if call.get("use_cache")]
    assert all("past_key_values" not in call for call in model.generate_calls)
    assert backend.kv_cache_hits == 0


def test_transformers_backend_tokenizes_under_the_model_lock() -> None:
    """Fast tokenizers are not thread-safe, so encode and decode hold the lock."""
    backend = _backend_with_stub(kv_cache=False)
    tokenizer = backend._tokenizer
    locked: list[bool] = []
    encode, decode = tokenizer.encode, tokenizer.decode

    def checked_encode(prompt: str, *, add_special_tokens: bool) -> list[int]:
        locked.append(backend._model_lock.locked())
        return encode(prompt, add_special_tokens=add_special_tokens)

    def checked_decode(token_ids: list[int], *, skip_special_tokens: bool) -> str:
        locked.append(backend._model_lock.locked())
        return decode(token_ids, skip_special_tokens=skip_special_tokens)

    tokenizer.encode = checked_encode
    tokenizer.decode = checked_decode
    asyncio.run(backend.generate(_snapshot("shared alpha"), GenerationConfig()))
    assert locked == [True, True]


def _fake_torch(cuda: bool, bf16: bool, mps: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: bf16),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize(
    ("cuda", "bf16", "dtype", "expected"),
    [
        (True, True, "auto", ("cuda", "bfloat16")),
        (True, False, "auto", ("cuda", "float16")),
        (False, False, "auto", ("cpu", "float32")),
        (True, True, "float32", ("cuda", "float32")),
        (False, False, "bfloat16", ("cpu", "bfloat16")),
    ],
)
def test_device_and_dtype_prefers_bfloat16(
    cuda: bool, bf16: bool, dtype: str, expected: tuple[str, str]
) -> None:
    torch = _fake_torch(cuda=cuda, bf16=bf16)
    assert transformers_text._device_and_dtype(torch, dtype) == expected


def test_transformers_backend_reads_dtype_from_backend_options() -> None:
    backend = transformers_text.TransformersTextBackend(
        backend_context(
            BackendConfig(
                kind="transformers",
                model_id="stub-model",
                options={"dtype": "float32"},
            )
        ),
    )
    assert backend.dtype == "float32"


@pytest.mark.parametrize(
    "options", [{"dtype": "fp8"}, {"dtype": "auto", "device_map": "auto"}]
)
def test_transformers_backend_rejects_invalid_backend_options(
    options: dict[str, object],
) -> None:
    with pytest.raises(ConfigError, match="transformers backend options"):
        transformers_text.TransformersTextBackend(
            backend_context(
                BackendConfig(
                    kind="transformers", model_id="stub-model", options=options
                )
            ),
        )


def test_transformers_backend_loads_the_configured_revision_and_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Auto:
        def __init__(self, kind: str, result: Any) -> None:
            self._kind, self._result = kind, result

        def from_pretrained(self, model_id: str, **kwargs: Any) -> Any:
            calls.append((self._kind, {"model_id": model_id, **kwargs}))
            return self._result

    fake = SimpleNamespace(
        AutoTokenizer=_Auto("tokenizer", _TokenizerStub()),
        AutoModelForCausalLM=_Auto("model", _ModelStub()),
    )
    monkeypatch.setattr(transformers_text, "_import_transformers", lambda: fake)
    backend = transformers_text.TransformersTextBackend(
        backend_context(
            BackendConfig(
                kind="transformers",
                model_id="org/model",
                options={"revision": "0123abcd", "dtype": "float32"},
            )
        ),
    )
    asyncio.run(backend.generate(_snapshot("shared alpha"), GenerationConfig()))

    assert calls == [
        ("tokenizer", {"model_id": "org/model", "revision": "0123abcd"}),
        (
            "model",
            {"model_id": "org/model", "revision": "0123abcd", "dtype": torch.float32},
        ),
    ]


def test_transformers_backend_returns_generated_token_logprobs() -> None:
    backend = _backend_with_stub()
    record = asyncio.run(backend.generate(_snapshot("alpha"), GenerationConfig()))

    assert record.logprobs == pytest.approx((-math.log(16),) * 2)
    assert backend._model.generate_calls[0]["output_logits"] is True
    assert isinstance(backend, SupportsLogprobs)
