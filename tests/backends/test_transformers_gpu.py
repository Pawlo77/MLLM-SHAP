"""Optional GPU integration tests for the transformers backend.

@see [[src/nlp_shap/backends/transformers/__init__.py]]
"""

import asyncio
import itertools
from pathlib import Path

import pytest

from nlp_shap import ExplainRunner, explain_config_from_yaml
from nlp_shap.backends._chat import snapshot_to_chat_messages
from nlp_shap.backends.transformers.chat import render_prompt
from nlp_shap.backends.transformers.text import TransformersTextBackend
from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"
"""Tiny random GPT-2 checkpoint; CPU friendly."""

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def transformers_backend_available() -> None:
    """Skip GPU tests when torch/transformers are unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


def _three_token_snapshot() -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text="one two three"),))
    return ConversationSnapshot.from_turns((turn,))


def test_transformers_runner_exact_reports_kv_cache_hits(
    transformers_backend_available: None,
) -> None:
    """A tiny exact run against HF reports non-zero prefix-cache hits."""
    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "dev_transformers.yaml"
    )
    output = ExplainRunner(
        explain_config_from_yaml(config_path.read_text(encoding="utf-8"))
    ).explain_sync(_three_token_snapshot())

    assert len(output.result.values) == 3
    assert output.metrics.executed > 0
    assert output.metrics.kv_cache_hits > 0
    assert output.metrics.kv_cache_reused_tokens > 0


def _coalition_prompts() -> list[str]:
    """Every coalition of five words, in an order where neighbours share prefixes."""
    words = ("one", "two", "three", "four", "five")
    prompts = []
    for mask in itertools.product((True, False), repeat=len(words)):
        kept = [word for word, present in zip(words, mask, strict=True) if present]
        prompts.append(" ".join(kept) or ".")
    return prompts


def _tiny_backend() -> TransformersTextBackend:
    return TransformersTextBackend(
        backend_context(
            BackendConfig(
                kind="transformers", model_id=TINY_MODEL, options={"dtype": "float32"}
            )
        )
    )


def _generate_all(kv_cache: bool) -> tuple[list[GenerationRecord], int, int]:
    backend = _tiny_backend()
    backend.set_kv_cache_enabled(kv_cache, 8)
    generation = GenerationConfig(max_new_tokens=8)

    async def run() -> list[GenerationRecord]:
        try:
            return [
                await backend.generate(_snapshot(prompt), generation)
                for prompt in _coalition_prompts()
            ]
        finally:
            await backend.aclose()

    records = asyncio.run(run())
    return records, backend.kv_cache_hits, backend.kv_cache_reused_tokens


def _snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def test_prefix_cache_does_not_change_greedy_outputs(
    transformers_backend_available: None,
) -> None:
    """Known truth: greedy generations are identical with the KV cache on and off."""
    cached, hits, reused = _generate_all(kv_cache=True)
    uncached, no_hits, no_reused = _generate_all(kv_cache=False)

    assert [r.text for r in cached] == [r.text for r in uncached]
    for with_cache, without in zip(cached, uncached, strict=True):
        assert with_cache.logprobs == pytest.approx(without.logprobs, abs=1e-4)
    assert hits > 0
    assert reused > hits
    assert (no_hits, no_reused) == (0, 0)


def test_logprobs_match_a_forward_pass_over_the_generated_tokens(
    transformers_backend_available: None,
) -> None:
    """Known truth: one logprob per generated token, equal to re-scoring them."""
    torch = pytest.importorskip("torch")
    backend = _tiny_backend()
    snapshot = _snapshot("one two three")
    record = asyncio.run(backend.generate(snapshot, GenerationConfig(max_new_tokens=8)))
    tokenizer, model = backend._tokenizer, backend._model
    prompt = render_prompt(tokenizer, snapshot_to_chat_messages(snapshot))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    with torch.no_grad():
        sequence = model.generate(
            input_ids=torch.tensor([prompt_ids], device=backend._device),
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
        logits = model(sequence[None, :-1]).logits[0].float()
    new_ids = sequence[len(prompt_ids) :]
    steps = logits[len(prompt_ids) - 1 :].log_softmax(dim=-1)
    expected = steps.gather(1, new_ids[:, None])[:, 0].tolist()

    assert record.logprobs is not None
    assert len(record.logprobs) == len(new_ids) >= 1
    assert all(value <= 0.0 for value in record.logprobs)
    assert record.logprobs == pytest.approx(expected, abs=1e-4)
