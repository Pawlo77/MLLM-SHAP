"""Tests for optional coalition progress callbacks.

@see [[src/mllm_shap/runtime/progress.py]]
"""

import asyncio

from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.pipeline.config import GenerationConfig
from mllm_shap.runtime.progress import NullCoalitionProgress
from mllm_shap.runtime.scheduler import CoalitionJob, InferenceScheduler
from mllm_shap.runtime.store import HotResultStore


def _job(key: str) -> CoalitionJob:
    snapshot = ConversationSnapshot(
        turns=(Turn(messages=(Message(role=Role.USER, text=key),)),),
        snapshot_id="s",
    )
    return CoalitionJob(
        coalition_key=key,
        snapshot_id="s",
        snapshot=snapshot,
        absence_policy="pad",
        mask_words=b"\x00",
        mask_n_bits=1,
        model_id="mock",
    )


async def _generate(snapshot: ConversationSnapshot) -> str:
    _ = snapshot
    return "ok"


def test_null_progress_is_safe_noop() -> None:
    progress = NullCoalitionProgress()
    progress.on_coalitions_planned(3)
    progress.on_coalition_finished(1, 3)


def test_scheduler_reports_planned_and_finished_progress(progress) -> None:
    jobs = (_job("a"), _job("b"), _job("c"))
    scheduler = InferenceScheduler(
        max_inflight=2,
        generation=GenerationConfig(),
        store=HotResultStore(maxsize=8),
        progress=progress,
    )
    metrics = asyncio.run(scheduler.run(jobs, _generate))
    assert metrics.executed == 3
    assert progress.planned == [3]
    assert progress.finished[-1] == (3, 3)
    assert len(progress.finished) == 3


def test_scheduler_progress_accumulates_across_runs(progress) -> None:
    """A second pass extends the planned total; finished counts keep growing."""
    scheduler = InferenceScheduler(
        max_inflight=1,
        generation=GenerationConfig(),
        store=HotResultStore(maxsize=8),
        progress=progress,
    )
    asyncio.run(scheduler.run((_job("a"), _job("b")), _generate))
    asyncio.run(scheduler.run((_job("c"),), _generate))
    assert progress.planned == [2, 3]
    assert [done for done, _ in progress.finished] == [1, 2, 3]
    assert progress.finished[-1] == (3, 3)
