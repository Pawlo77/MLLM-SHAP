"""Optional GPU integration tests for the transformers backend.

@see [[src/mllm_shap/backends/transformers/__init__.py]]
"""

import asyncio
import itertools
from pathlib import Path

import pytest

from mllm_shap import ExplainRunner, explain_config_from_yaml
from mllm_shap.backends._chat import snapshot_to_chat_messages
from mllm_shap.backends.transformers.chat import render_prompt
from mllm_shap.backends.transformers.text import TransformersTextBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from mllm_shap.pipeline.config import BackendConfig
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


def test_transformers_backend_counts_sequences_tokens_and_time(
    transformers_backend_available: None,
) -> None:
    """Generation and forcing add to the cumulative cost counters."""
    backend = _tiny_backend()
    backend.set_kv_cache_enabled(False, 8)
    snapshot = _three_token_snapshot()
    assert backend.cost_counters.forward_sequences == 0

    record = asyncio.run(backend.generate(snapshot, GenerationConfig(max_new_tokens=5)))
    after_generation = backend.cost_counters
    assert after_generation.forward_sequences == 1
    assert after_generation.prompt_tokens > 0
    assert 0 < after_generation.generated_tokens <= 5
    assert after_generation.device_seconds > 0.0
    assert record.text is not None

    asyncio.run(backend.score_continuations(snapshot, ["one two", "three"]))
    after_forcing = backend.cost_counters - after_generation
    assert after_forcing.forward_sequences == 2
    assert after_forcing.generated_tokens == 0
    assert after_forcing.prompt_tokens > 2 * after_generation.prompt_tokens - 1
    assert after_forcing.device_seconds > 0.0


def _batched_backend(batch_size: int) -> TransformersTextBackend:
    return TransformersTextBackend(
        backend_context(
            BackendConfig(
                kind="transformers",
                model_id=TINY_MODEL,
                options={"dtype": "float32", "batch_size": batch_size},
            )
        )
    )


def _prompt_snapshots() -> list[ConversationSnapshot]:
    return [
        ConversationSnapshot.from_turns((
            Turn(messages=(Message(role=Role.USER, text=prompt),)),
        ))
        for prompt in _coalition_prompts()[:12]
    ]


def test_batched_forced_scores_equal_the_one_prompt_scores(
    transformers_backend_available: None,
) -> None:
    """Concurrent scoring in padded batches matches unbatched scoring to 1e-4."""
    snapshots = _prompt_snapshots()
    continuations = ["one two", "three four five"]
    single = _batched_backend(1)

    async def reference() -> list[list]:
        return [
            await single.score_continuations(s, continuations, 5) for s in snapshots
        ]

    async def batched(backend: TransformersTextBackend) -> list[list]:
        return list(
            await asyncio.gather(*[
                backend.score_continuations(s, continuations, 5) for s in snapshots
            ])
        )

    expected = asyncio.run(reference())
    backend = _batched_backend(8)
    got = asyncio.run(batched(backend))
    for want_row, got_row in zip(expected, got, strict=True):
        for want, have in zip(want_row, got_row, strict=True):
            assert have.token_logprobs == pytest.approx(want.token_logprobs, abs=1e-4)
            assert have.top_ids == want.top_ids
    assert (
        backend.cost_counters.forward_sequences
        == single.cost_counters.forward_sequences
    )
    assert backend.batch_size == 8


def test_a_batch_error_reaches_every_waiting_caller(
    transformers_backend_available: None,
) -> None:
    """A failing forward pass raises in each queued call instead of hanging."""
    backend = _batched_backend(4)
    snapshots = _prompt_snapshots()[:3]

    def boom(batch: list) -> list:
        msg = "forward failed"
        raise RuntimeError(msg)

    async def run() -> list:
        await backend._ensure_loaded()
        backend._force_batch_sync = boom
        return await asyncio.gather(
            *[backend.score_continuations(s, ["one"]) for s in snapshots],
            return_exceptions=True,
        )

    results = asyncio.run(run())
    assert all(isinstance(r, RuntimeError) for r in results)


def _all_coalition_snapshots(words: int) -> list[ConversationSnapshot]:
    names = [f"word{i}" for i in range(words)]
    snapshots = []
    for mask in itertools.product([False, True], repeat=words):
        text = " ".join(n for n, keep in zip(names, mask, strict=True) if keep) or "."
        snapshots.append(
            ConversationSnapshot.from_turns((
                Turn(messages=(Message(role=Role.USER, text=text),)),
            ))
        )
    return snapshots


def test_batched_greedy_generation_equals_unbatched_on_every_coalition(
    transformers_backend_available: None,
) -> None:
    """200+ coalitions of eight words give the same greedy text and logprobs."""
    snapshots = _all_coalition_snapshots(8)
    generation = GenerationConfig(max_new_tokens=6)
    single = _batched_backend(1)
    single.set_kv_cache_enabled(False, 8)
    batched = _batched_backend(16)

    async def reference() -> list[GenerationRecord]:
        return [await single.generate(s, generation) for s in snapshots]

    async def together() -> list[GenerationRecord]:
        return list(
            await asyncio.gather(*[batched.generate(s, generation) for s in snapshots])
        )

    expected = asyncio.run(reference())
    got = asyncio.run(together())
    assert len(got) == len(expected) == 256
    assert [r.text for r in got] == [r.text for r in expected]
    for want, have in zip(expected, got, strict=True):
        assert have.logprobs is not None
        assert want.logprobs is not None
        assert have.logprobs == pytest.approx(want.logprobs, abs=1e-3)
    assert batched.cost_counters.forward_sequences == 256
