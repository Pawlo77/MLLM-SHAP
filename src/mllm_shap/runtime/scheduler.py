"""Async coalition inference scheduling with deduplication.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_scheduler.py]]
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass

from ..domain.config import GenerationConfig
from ..domain.conversation import ConversationSnapshot
from ..domain.generation import GenerationRecord
from ..masking.codec import PackedMask
from .archive import CoalitionRecordDraft, RunArchive
from .dedup import CoalitionDedupRegistry
from .progress import CoalitionProgress, NullCoalitionProgress
from .store import HotResultStore

GenerateFn = Callable[[ConversationSnapshot], Awaitable[str | GenerationRecord]]
"""Async callable that returns the generated text, or a record carrying it and
its token logprobs, for one snapshot."""


def _first_error(group: BaseExceptionGroup[BaseException]) -> BaseException:
    """Return the first leaf exception of ``group`` that is not a cancellation.

    Nested groups are searched depth first; ``group`` itself is returned when it
    holds cancellations only.
    """
    return _find_error(group) or group


def _find_error(group: BaseExceptionGroup[BaseException]) -> BaseException | None:
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            found = _find_error(error)
            if found is not None:
                return found
        elif not isinstance(error, asyncio.CancelledError):
            return error
    return None


@dataclass(frozen=True, slots=True)
class SchedulerMetrics:
    """Counters collected while executing coalition jobs."""

    requested: int
    """Number of coalition jobs submitted to the scheduler."""

    executed: int
    """Number of jobs that invoked the backend generate callable."""

    deduplicated: int
    """Number of jobs skipped because the coalition key was already executed."""

    cache_hits: int
    """Number of jobs served from the hot result store without backend calls."""

    kv_cache_hits: int = 0
    """Coalition generations that reused a cached prompt prefix; filled by the
    orchestrator from the backend, 0 from the scheduler alone."""

    kv_cache_reused_tokens: int = 0
    """Prompt tokens served from cached prefixes; filled like ``kv_cache_hits``."""


@dataclass(frozen=True, slots=True)
class CoalitionJob:
    """One coalition evaluation request routed through the scheduler."""

    coalition_key: str
    """Stable deduplication key for the coalition."""

    snapshot_id: str
    """Identifier of the evaluated conversation snapshot."""

    snapshot: ConversationSnapshot
    """Conversation snapshot passed to the backend for generation."""

    absence_policy: str
    """Registered absence-policy identifier used for rendering."""

    mask_words: bytes
    """Packed coalition mask bytes."""

    mask_n_bits: int
    """Original coalition mask bit length."""

    model_id: str
    """Backend model identifier used for generation."""

    prefix_hash: str = ""
    """Shared prompt prefix hash used for KV-cache grouping."""


class InferenceScheduler:
    """Execute coalition jobs with bounded concurrency and optional deduplication."""

    def __init__(
        self,
        max_inflight: int,
        generation: GenerationConfig,
        store: HotResultStore,
        dedup: CoalitionDedupRegistry | None = None,
        archive: RunArchive | None = None,
        pending_limit: int | None = None,
        progress: CoalitionProgress | None = None,
    ) -> None:
        """Configure concurrency, caching, archiving and progress reporting."""
        if max_inflight < 1:
            msg = "max_inflight must be at least 1"
            raise ValueError(msg)
        self._max_inflight = max_inflight
        self._generation = generation
        self._store = store
        self._dedup = dedup
        self._archive = archive
        self._semaphore = asyncio.Semaphore(max_inflight)
        self._pending_limit = pending_limit or max(max_inflight * 4, max_inflight)
        self._progress: CoalitionProgress = progress or NullCoalitionProgress()
        # Progress counters span every run of this scheduler, so a second pass
        # (Neyman phase two) extends the plan instead of restarting it.
        self._planned = 0
        self._finished = 0

    async def run(
        self,
        jobs: Sequence[CoalitionJob],
        generate: GenerateFn,
    ) -> SchedulerMetrics:
        """Execute coalition jobs and return scheduler counters."""
        return await self.run_iter(iter(jobs), generate, total=len(jobs))

    async def run_iter(
        self,
        jobs: Iterable[CoalitionJob],
        generate: GenerateFn,
        total: int | None = None,
    ) -> SchedulerMetrics:
        """Execute jobs from an iterable without creating all coroutines upfront.

        Jobs run in one :class:`asyncio.TaskGroup`, with at most
        ``pending_limit`` tasks alive at a time. The first job failure cancels
        every other task and is re-raised as is (not wrapped in an exception
        group), so callers never observe partial success. With ``total`` the
        progress sink is told the cumulative planned count of all runs so far.
        """
        if total is not None:
            self._planned += total
            self._progress.on_coalitions_planned(self._planned)
        mutable = {
            "executed": 0,
            "deduplicated": 0,
            "cache_hits": 0,
            "requested": 0,
        }
        slots = asyncio.Semaphore(self._pending_limit)

        def release_slot(_task: asyncio.Task[None]) -> None:
            slots.release()

        try:
            async with asyncio.TaskGroup() as group:
                for job in jobs:
                    mutable["requested"] += 1
                    if total is None:
                        self._planned += 1
                    await slots.acquire()
                    task = group.create_task(self._process_job(job, generate, mutable))
                    task.add_done_callback(release_slot)
        except BaseExceptionGroup as errors:
            raise _first_error(errors) from None

        return SchedulerMetrics(
            requested=mutable["requested"],
            executed=mutable["executed"],
            deduplicated=mutable["deduplicated"],
            cache_hits=mutable["cache_hits"],
        )

    async def _process_job(
        self,
        job: CoalitionJob,
        generate: GenerateFn,
        mutable: dict[str, int],
    ) -> None:
        try:
            cached = self._store.get(job.coalition_key)
            if cached is not None:
                mutable["cache_hits"] += 1
                logprobs = self._store.logprobs(job.coalition_key)
                self._maybe_archive(job, cached, logprobs, True, elapsed_ms=0.0)
                return

            if self._dedup is not None and not self._dedup.observe(job.coalition_key):
                mutable["deduplicated"] += 1
                return

            async with self._semaphore:
                started = time.perf_counter()
                generated = await generate(job.snapshot)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            mutable["executed"] += 1
            text, logprobs = (
                (generated, None)
                if isinstance(generated, str)
                else (generated.text, generated.logprobs)
            )
            self._store.put(job.coalition_key, text, logprobs)
            self._maybe_archive(job, text, logprobs, False, elapsed_ms=elapsed_ms)
        finally:
            self._finished += 1
            self._progress.on_coalition_finished(
                self._finished, max(self._planned, self._finished)
            )

    def _maybe_archive(
        self,
        job: CoalitionJob,
        generation_text: str,
        logprobs: tuple[float, ...] | None,
        cache_hit: bool,
        elapsed_ms: float,
    ) -> None:
        if self._archive is None:
            return
        self._archive.append(
            CoalitionRecordDraft(
                snapshot_id=job.snapshot_id,
                coalition_key=job.coalition_key,
                mask=PackedMask(words=job.mask_words, n_bits=job.mask_n_bits),
                absence_policy=job.absence_policy,
                model_id=job.model_id,
                generation_text=generation_text,
                elapsed_ms=elapsed_ms,
                cache_hit=cache_hit,
                logprobs=logprobs,
            )
        )
