"""Batching queue of the transformers backend, with fake batch functions (no torch).

@see [[src/mllm_shap/backends/transformers/text.py]]
"""

import asyncio
from typing import Any

import pytest

from mllm_shap.backends.transformers.options import parse_transformers_options
from mllm_shap.backends.transformers.text import TransformersTextBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import ForcedScore, GenerationRecord
from mllm_shap.errors import ConfigError
from mllm_shap.pipeline.config import BackendConfig
from tests.backends._context import backend_context


def _snapshot(text: str) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _backend(batch_size: int, wait_ms: float = 1.0) -> TransformersTextBackend:
    backend = TransformersTextBackend(
        backend_context(
            BackendConfig(
                kind="transformers",
                model_id="unused",
                options={"batch_size": batch_size, "batch_wait_ms": wait_ms},
            )
        )
    )

    async def loaded() -> None:
        return None

    backend._ensure_loaded = loaded  # type: ignore[method-assign]
    return backend


def test_options_validate_the_batch_settings() -> None:
    assert parse_transformers_options({}).batch_size == 1
    with pytest.raises(ConfigError):
        parse_transformers_options({"batch_size": 0})
    with pytest.raises(ConfigError):
        parse_transformers_options({"batch_wait_ms": -1.0})
    assert _backend(8).batch_size == 8


def test_concurrent_forced_scores_run_as_one_full_batch() -> None:
    backend = _backend(4)
    batches: list[int] = []

    def fake(batch: list[Any]) -> list[list[ForcedScore]]:
        batches.append(len(batch))
        return [[ForcedScore((float(i),))] for i, _ in enumerate(batch)]

    backend._force_batch_sync = fake  # type: ignore[method-assign]

    async def run() -> list[list[ForcedScore]]:
        return list(
            await asyncio.gather(*[
                backend.score_continuations(_snapshot(f"w{i}"), ["x"]) for i in range(4)
            ])
        )

    results = asyncio.run(run())
    assert batches == [4]
    assert [r[0].token_logprobs for r in results] == [(0.0,), (1.0,), (2.0,), (3.0,)]


def test_a_partial_batch_runs_after_the_wait() -> None:
    backend = _backend(8, wait_ms=2.0)
    batches: list[int] = []

    def fake(batch: list[Any]) -> list[list[ForcedScore]]:
        batches.append(len(batch))
        return [[ForcedScore(())] for _ in batch]

    backend._force_batch_sync = fake  # type: ignore[method-assign]

    async def run() -> None:
        await asyncio.gather(*[
            backend.score_continuations(_snapshot("a"), ["x"]) for _ in range(3)
        ])

    asyncio.run(run())
    assert batches == [3]


def test_a_failing_batch_raises_in_every_caller() -> None:
    backend = _backend(3)

    def boom(batch: list[Any]) -> list[list[ForcedScore]]:
        msg = "forward failed"
        raise RuntimeError(msg)

    backend._force_batch_sync = boom  # type: ignore[method-assign]

    async def run() -> list[Any]:
        return await asyncio.gather(
            *[backend.score_continuations(_snapshot("a"), ["x"]) for _ in range(3)],
            return_exceptions=True,
        )

    assert all(isinstance(r, RuntimeError) for r in asyncio.run(run()))


def test_greedy_generations_are_batched_and_sampled_ones_are_not() -> None:
    backend = _backend(4)
    batches: list[int] = []

    def fake_batch(batch: list[Any]) -> list[GenerationRecord]:
        batches.append(len(batch))
        return [GenerationRecord(text=f"b{i}") for i, _ in enumerate(batch)]

    def fake_single(snapshot: Any, generation: Any) -> GenerationRecord:
        return GenerationRecord(text="single")

    backend._generate_batch_sync = fake_batch  # type: ignore[method-assign]
    backend._generate_sync = fake_single  # type: ignore[method-assign]
    greedy = GenerationConfig(max_new_tokens=4)
    sampled = GenerationConfig(max_new_tokens=4, temperature=0.7, top_k=5)

    async def run() -> tuple[list[GenerationRecord], GenerationRecord]:
        many = await asyncio.gather(*[
            backend.generate(_snapshot(f"w{i}"), greedy) for i in range(4)
        ])
        return list(many), await backend.generate(_snapshot("z"), sampled)

    many, alone = asyncio.run(run())
    assert batches == [4]
    assert [r.text for r in many] == ["b0", "b1", "b2", "b3"]
    assert alone.text == "single"


def test_batch_size_one_scores_each_prompt_on_its_own() -> None:
    backend = _backend(1)
    seen: list[tuple[str, ...]] = []

    def fake(snapshot: Any, continuations: tuple[str, ...], top_k: int) -> list[Any]:
        seen.append(continuations)
        return [ForcedScore(())]

    backend._force_sync = fake  # type: ignore[method-assign]
    asyncio.run(backend.score_continuations(_snapshot("a"), ["x", "y"]))
    assert seen == [("x", "y")]


def test_the_cost_counters_accumulate() -> None:
    backend = _backend(1)
    backend._count(10, 3, 1, 0.5)
    backend._count(5, 0, 2, 0.25)
    counters = backend.cost_counters
    assert (counters.forward_sequences, counters.prompt_tokens) == (3, 15)
    assert (counters.generated_tokens, counters.device_seconds) == (3, 0.75)
