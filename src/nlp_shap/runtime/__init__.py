"""Runtime archive, deduplication, caching, and scheduling.

@domain runtime
@see [[knowledge/architecture/runtime]]
"""

from .archive import (
    BASE_GENERATION_FILE,
    CoalitionRecord,
    CoalitionRecordDraft,
    RunArchive,
)
from .dedup import CoalitionDedupRegistry, build_coalition_key, dedup_enabled
from .kv_cache import (
    PrefixCacheManager,
    build_snapshot_prefix_hash,
    group_jobs_for_prefix_cache,
)
from .metrics import PerfSummary
from .progress import CoalitionProgress, NullCoalitionProgress
from .scheduler import CoalitionJob, InferenceScheduler, SchedulerMetrics
from .store import HotResultStore
from .telemetry import (
    InMemoryObservabilitySink,
    NullObservabilitySink,
    ObservabilitySink,
    SpanRecord,
)

__all__ = [
    "BASE_GENERATION_FILE",
    "CoalitionDedupRegistry",
    "CoalitionJob",
    "CoalitionProgress",
    "CoalitionRecord",
    "CoalitionRecordDraft",
    "HotResultStore",
    "InMemoryObservabilitySink",
    "InferenceScheduler",
    "NullCoalitionProgress",
    "NullObservabilitySink",
    "ObservabilitySink",
    "PerfSummary",
    "PrefixCacheManager",
    "RunArchive",
    "SchedulerMetrics",
    "SpanRecord",
    "build_coalition_key",
    "build_snapshot_prefix_hash",
    "dedup_enabled",
    "group_jobs_for_prefix_cache",
]
