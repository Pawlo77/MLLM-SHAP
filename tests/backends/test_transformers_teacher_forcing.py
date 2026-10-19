"""Teacher-forced scoring of the transformers backend against a plain forward pass.

@see [[src/mllm_shap/backends/transformers/text.py]]
"""

import asyncio
import math

import pytest

from mllm_shap.backends.transformers import text as transformers_text
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.pipeline.config import BackendConfig
from mllm_shap.protocols import SupportsTeacherForcing
from tests.backends._context import backend_context

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


class _CharTokenizer:
    """Byte-level tokenizer: ids are ``ord(c) % 32``; no chat template."""

    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        _ = add_special_tokens
        return [ord(c) % 32 for c in text]


def _backend() -> tuple[transformers_text.TransformersTextBackend, torch.nn.Module]:
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=32)
    ).eval()
    backend = transformers_text.TransformersTextBackend(
        backend_context(BackendConfig(kind="transformers", model_id="tiny"))
    )
    backend._model = model
    backend._tokenizer = _CharTokenizer()
    backend._device = torch.device("cpu")
    return backend, model


def _snapshot() -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn((Message(role=Role.USER, text="hello there"),)),
    ))


def test_forced_logprobs_match_a_hand_computed_forward_pass() -> None:
    backend, model = _backend()
    prompt = [ord(c) % 32 for c in "user: hello there"]
    answer = [ord(c) % 32 for c in "yes no"]

    with torch.no_grad():
        logits = model(torch.tensor([prompt + answer])).logits[0].double()
    expected = [
        float(logits[len(prompt) - 1 + t].log_softmax(-1)[token])
        for t, token in enumerate(answer)
    ]

    (forced,) = asyncio.run(backend.score_continuations(_snapshot(), ["yes no"]))
    assert isinstance(backend, SupportsTeacherForcing)
    assert forced.token_logprobs == pytest.approx(expected, abs=1e-5)
    assert forced.top_ids == ()


def test_top_k_lists_the_most_probable_next_tokens() -> None:
    backend, _ = _backend()
    (forced,) = asyncio.run(backend.score_continuations(_snapshot(), ["yes"], top_k=5))

    assert len(forced.top_ids) == len(forced.token_logprobs) == 3
    for ids, logs in zip(forced.top_ids, forced.top_logprobs, strict=True):
        assert len(ids) == 5
        assert list(logs) == sorted(logs, reverse=True)
        assert math.fsum(math.exp(v) for v in logs) <= 1.0 + 1e-6
    # Full vocabulary: the distribution sums to one and holds the forced token.
    (full,) = asyncio.run(backend.score_continuations(_snapshot(), ["a"], top_k=99))
    assert math.fsum(math.exp(v) for v in full.top_logprobs[0]) == pytest.approx(1.0)
    assert full.token_logprobs[0] in full.top_logprobs[0]


def test_an_empty_continuation_scores_nothing_and_audio_is_refused() -> None:
    backend, _ = _backend()
    (empty,) = asyncio.run(backend.score_continuations(_snapshot(), [""]))
    assert empty.token_logprobs == ()

    spoken = ConversationSnapshot.from_turns((
        Turn((
            Message(
                role=Role.USER,
                content=(AudioPart(source=b"RIFF", sample_rate=16000),),
            ),
        )),
    ))
    with pytest.raises(ValueError, match="text prompts"):
        asyncio.run(backend.score_continuations(spoken, ["x"]))
