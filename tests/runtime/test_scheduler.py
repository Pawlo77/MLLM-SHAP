"""Tests for async coalition inference scheduling.

@see [[src/nlp_shap/runtime/scheduler.py]]
"""

import asyncio
import warnings
from collections.abc import Iterable
from pathlib import Path

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.config import GenerationConfig
from nlp_shap.runtime.archive import RunArchive
from nlp_shap.runtime.dedup import CoalitionDedupRegistry, build_coalition_key
from nlp_shap.runtime.scheduler import (
    CoalitionJob,
    InferenceScheduler,
    SchedulerMetrics,
    _first_error,
)
from nlp_shap.runtime.store import HotResultStore


async def _slow_generate(snapshot: ConversationSnapshot) -> str:
    await asyncio.sleep(0.01)
    return f"generated:{snapshot.snapshot_id}"


def _make_job(
    snapshot: ConversationSnapshot,
    coalition_key: str,
    mask_present: tuple[bool, ...] = (True, False, True),
) -> CoalitionJob:
    generation = GenerationConfig(temperature=0.0)
    key = coalition_key or build_coalition_key(
        snapshot_id=snapshot.snapshot_id,
        player_ids=("p0", "p1", "p2"),
        mask_present=mask_present,
        absence_policy="delete",
        model_id="mock",
        generation=generation,
    )
    packed = MaskCodec.pack(mask_present)
    return CoalitionJob(
        coalition_key=key,
        snapshot_id=snapshot.snapshot_id,
        snapshot=snapshot,
        absence_policy="delete",
        mask_words=packed.words,
        mask_n_bits=packed.n_bits,
        model_id="mock",
    )


def test_scheduler_run_iter_accepts_job_generator(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """run_iter streams jobs without materializing the full job list."""

    def job_stream() -> Iterable[CoalitionJob]:
        for index in range(100):
            yield _make_job(sample_snapshot, coalition_key=f"stream-{index}")

    async def run() -> SchedulerMetrics:
        scheduler = InferenceScheduler(
            max_inflight=2,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=None,
            pending_limit=4,
        )
        return await scheduler.run_iter(job_stream(), _slow_generate)

    metrics = asyncio.run(run())
    assert metrics.requested == 100
    assert metrics.executed == 100


def test_scheduler_bounds_inflight(sample_snapshot: ConversationSnapshot) -> None:
    """Concurrent backend calls never exceed max_inflight."""
    max_inflight = 2
    concurrent = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def tracked_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal concurrent, max_seen
        async with lock:
            concurrent += 1
            max_seen = max(max_seen, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        return f"text:{snapshot.snapshot_id}"

    async def run() -> int:
        scheduler = InferenceScheduler(
            max_inflight=max_inflight,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=None,
        )
        jobs = [
            _make_job(sample_snapshot, coalition_key=f"unique-{index}")
            for index in range(100)
        ]
        metrics = await scheduler.run(jobs, tracked_generate)
        return max_seen, metrics.executed

    max_seen, executed = asyncio.run(run())
    assert max_seen <= max_inflight
    assert executed == 100


def test_scheduler_reuses_store_for_repeated_masks(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """One hundred jobs with ten unique coalition keys execute ten backend calls."""
    executed = 0

    async def counting_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal executed
        executed += 1
        return "ok"

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=4,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=CoalitionDedupRegistry(),
        )
        jobs = [
            _make_job(sample_snapshot, coalition_key=f"key-{index % 10}")
            for index in range(100)
        ]
        metrics = await scheduler.run(jobs, counting_generate)
        assert metrics.requested == 100
        assert metrics.executed == 10
        assert metrics.cache_hits == 90
        assert metrics.deduplicated == 0

    asyncio.run(run())
    assert executed == 10


def test_scheduler_counts_deduplicated_inflight_duplicates(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Concurrent duplicate keys deduplicate before the hot store is populated."""
    started = asyncio.Event()
    release = asyncio.Event()
    executed = 0

    async def gated_generate(snapshot: ConversationSnapshot) -> str:
        nonlocal executed
        executed += 1
        started.set()
        await release.wait()
        return "ok"

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=4,
            generation=GenerationConfig(temperature=0.0),
            store=HotResultStore(),
            dedup=CoalitionDedupRegistry(),
        )
        key = "shared-key"
        jobs = [_make_job(sample_snapshot, coalition_key=key) for _ in range(3)]
        task = asyncio.create_task(scheduler.run(jobs, gated_generate))
        await started.wait()
        release.set()
        metrics = await task
        assert metrics.executed == 1
        assert metrics.deduplicated == 2

    asyncio.run(run())


def test_scheduler_uses_hot_store_before_backend(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Cached coalition results avoid backend execution."""
    store = HotResultStore()
    key = build_coalition_key(
        snapshot_id=sample_snapshot.snapshot_id,
        player_ids=("p0",),
        mask_present=(True,),
        absence_policy="delete",
        model_id="mock",
        generation=GenerationConfig(temperature=0.0),
    )
    store.put(key, "cached")

    async def run() -> None:
        scheduler = InferenceScheduler(
            max_inflight=2,
            generation=GenerationConfig(temperature=0.0),
            store=store,
            dedup=CoalitionDedupRegistry(),
        )
        jobs = [_make_job(sample_snapshot, coalition_key=key)]
        metrics = await scheduler.run(jobs, _slow_generate)
        assert metrics.cache_hits == 1
        assert metrics.executed == 0

    asyncio.run(run())


def test_scheduler_appends_archive_records(
    tmp_path: Path,
    sample_snapshot: ConversationSnapshot,
    run_manifest,
) -> None:
    """Executed jobs append coalition rows to the run archive."""
    root = tmp_path / "run-scheduler"

    async def run() -> None:
        with RunArchive.open(root, run_manifest) as archive:
            scheduler = InferenceScheduler(
                max_inflight=2,
                generation=GenerationConfig(temperature=0.0),
                store=HotResultStore(),
                dedup=None,
                archive=archive,
            )
            jobs = [_make_job(sample_snapshot, coalition_key="archive-key")]
            await scheduler.run(jobs, _slow_generate)

    asyncio.run(run())

    with RunArchive.load(root) as archive:
        records = list(archive.history_lazy())
    assert len(records) == 1
    assert records[0].generation_text.startswith("generated:")


def _job(index: int, snapshot: ConversationSnapshot) -> CoalitionJob:
    return CoalitionJob(
        coalition_key=f"key-{index}",
        snapshot_id="snap",
        snapshot=snapshot,
        absence_policy="pad",
        mask_words=b"\x00",
        mask_n_bits=1,
        model_id="mock",
    )


def test_scheduler_raises_first_failure_and_cancels_rest(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """A failure on an early job must surface from run(), not be swallowed."""
    calls: list[str] = []

    async def generate(snapshot: ConversationSnapshot) -> str:
        calls.append("x")
        if len(calls) == 3:
            msg = "backend exploded"
            raise RuntimeError(msg)
        await asyncio.sleep(0.001)
        return "ok"

    scheduler = InferenceScheduler(
        max_inflight=2, generation=GenerationConfig(), store=HotResultStore(maxsize=64)
    )
    jobs = [_job(i, sample_snapshot) for i in range(20)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeError, match="backend exploded"):
            asyncio.run(scheduler.run(jobs, generate))
    assert len(calls) < 20


def test_scheduler_unwraps_concurrent_failures(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """Several jobs failing together raise one plain exception, not a group."""

    async def generate(snapshot: ConversationSnapshot) -> str:
        await asyncio.sleep(0)
        msg = "every call fails"
        raise ValueError(msg)

    scheduler = InferenceScheduler(
        max_inflight=4, generation=GenerationConfig(), store=HotResultStore()
    )
    jobs = [_job(i, sample_snapshot) for i in range(8)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="every call fails") as caught:
            asyncio.run(scheduler.run(jobs, generate))
    assert not isinstance(caught.value, BaseExceptionGroup)


def test_scheduler_propagates_job_iterator_failure(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """An exception raised while producing jobs cancels running jobs and surfaces."""
    started: list[int] = []

    async def generate(snapshot: ConversationSnapshot) -> str:
        started.append(1)
        await asyncio.sleep(10)
        return "never"

    def broken_jobs() -> Iterable[CoalitionJob]:
        yield _job(0, sample_snapshot)
        msg = "job rendering failed"
        raise KeyError(msg)

    scheduler = InferenceScheduler(
        max_inflight=2, generation=GenerationConfig(), store=HotResultStore()
    )
    with pytest.raises(KeyError, match="job rendering failed"):
        asyncio.run(scheduler.run_iter(broken_jobs(), generate))


def test_scheduler_bounds_live_tasks_by_pending_limit(
    sample_snapshot: ConversationSnapshot,
) -> None:
    """At most ``pending_limit`` jobs are pulled from the iterator ahead of work."""
    produced = 0
    finished = 0
    max_ahead = 0

    def counting_jobs() -> Iterable[CoalitionJob]:
        nonlocal produced, max_ahead
        for index in range(50):
            produced += 1
            max_ahead = max(max_ahead, produced - finished)
            yield _job(index, sample_snapshot)

    async def generate(snapshot: ConversationSnapshot) -> str:
        nonlocal finished
        await asyncio.sleep(0.001)
        finished += 1
        return "ok"

    scheduler = InferenceScheduler(
        max_inflight=2,
        generation=GenerationConfig(),
        store=HotResultStore(maxsize=64),
        pending_limit=3,
    )
    metrics = asyncio.run(scheduler.run_iter(counting_jobs(), generate))
    assert metrics.executed == 50
    assert max_ahead <= 4  # three live tasks plus the job being produced


def test_first_error_skips_nested_groups_of_cancellations() -> None:
    failure = RuntimeError("backend crashed")
    group = BaseExceptionGroup(
        "run",
        [
            BaseExceptionGroup("inner", [asyncio.CancelledError()]),
            failure,
        ],
    )
    assert _first_error(group) is failure


def test_first_error_returns_group_when_everything_was_cancelled() -> None:
    group = BaseExceptionGroup("run", [asyncio.CancelledError()])
    assert _first_error(group) is group


def test_scheduler_keeps_logprobs_for_store_and_archive(
    tmp_path: Path,
    sample_snapshot: ConversationSnapshot,
    run_manifest,
) -> None:
    """Records with logprobs reach the store and every archived row, hits too."""
    root = tmp_path / "run-logprobs"
    store = HotResultStore()

    async def generate(snapshot: ConversationSnapshot) -> GenerationRecord:
        return GenerationRecord(text="out", logprobs=(-0.5, -0.25))

    async def run() -> None:
        with RunArchive.open(root, run_manifest) as archive:
            scheduler = InferenceScheduler(
                max_inflight=1,
                generation=GenerationConfig(),
                store=store,
                archive=archive,
            )
            job = _make_job(sample_snapshot, coalition_key="lp-key")
            await scheduler.run([job], generate)
            await scheduler.run([job], generate)

    asyncio.run(run())

    assert store.get("lp-key") == "out"
    assert store.logprobs("lp-key") == (-0.5, -0.25)
    with RunArchive.load(root) as archive:
        rows = list(archive.history_lazy())
    assert [(row.cache_hit, row.logprobs) for row in rows] == [
        (False, (-0.5, -0.25)),
        (True, (-0.5, -0.25)),
    ]
