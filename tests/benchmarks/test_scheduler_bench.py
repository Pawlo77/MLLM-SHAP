"""Benchmarks for the coalition scheduler.

@see [[src/mllm_shap/runtime/scheduler.py]]
"""

import asyncio
import tracemalloc
from collections.abc import Iterator

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.runtime.scheduler import (
    CoalitionJob,
    InferenceScheduler,
    SchedulerMetrics,
)
from mllm_shap.runtime.store import HotResultStore

_SNAPSHOT = ConversationSnapshot.from_turns((
    Turn(messages=(Message(role=Role.USER, text="bench prompt"),)),
))


def _jobs(count: int) -> Iterator[CoalitionJob]:
    for index in range(count):
        yield CoalitionJob(
            coalition_key=f"key-{index}",
            snapshot_id=_SNAPSHOT.snapshot_id,
            snapshot=_SNAPSHOT,
            absence_policy="pad",
            mask_words=b"\x00",
            mask_n_bits=1,
            model_id="mock",
        )


async def _noop_generate(snapshot: ConversationSnapshot) -> str:
    return snapshot.snapshot_id


def _run_jobs(count: int) -> SchedulerMetrics:
    scheduler = InferenceScheduler(
        max_inflight=4,
        generation=GenerationConfig(),
        store=HotResultStore(maxsize=count),
    )
    return asyncio.run(scheduler.run_iter(_jobs(count), _noop_generate, total=count))


@pytest.mark.bench
def test_scheduler_1000_noop_jobs(benchmark: BenchmarkFixture) -> None:
    """Scheduling overhead of 1 000 jobs at ``max_inflight=4``."""
    metrics = benchmark(_run_jobs, 1000)

    assert metrics.executed == 1000


@pytest.mark.bench
def test_scheduler_streaming_memory_probe() -> None:
    """Streaming 10k jobs keeps the traced peak far below one job list's size.

    A probe, not a gate on allocator details: it records the peak in the
    benchmark output via ``print`` and fails only on gross growth.
    """
    tracemalloc.start()
    try:
        metrics = _run_jobs(10_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    print(f"scheduler 10k jobs traced peak: {peak / 1024:.0f} KiB")
    assert metrics.executed == 10_000
    assert peak < 64 * 1024 * 1024
