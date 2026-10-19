---
title: "runtime"
type: module
domain: runtime
parent: "[[mllm-shap]]"
tags: [architecture, runtime]
created: 2026-10-07
updated: 2026-10-18
related-code:
  - "[[src/mllm_shap/runtime/__init__.py]]"
  - "[[src/mllm_shap/runtime/archive.py]]"
  - "[[src/mllm_shap/runtime/dedup.py]]"
  - "[[src/mllm_shap/runtime/kv_cache.py]]"
  - "[[src/mllm_shap/runtime/metrics.py]]"
  - "[[src/mllm_shap/runtime/progress.py]]"
  - "[[src/mllm_shap/runtime/scheduler.py]]"
  - "[[src/mllm_shap/runtime/store.py]]"
  - "[[src/mllm_shap/runtime/telemetry.py]]"
---
# runtime

## Responsibility

- Execute coalition jobs concurrently, skip duplicates, keep generated text for scoring, persist the [[Run archive]], cache prompt prefixes for backends that can reuse them, and collect timings.

## Public surface

- `InferenceScheduler(max_inflight, generation, store, dedup, archive, pending_limit, progress)`: `run(jobs, generate)` and `run_iter(jobs, generate, total=None)` consume a job iterator lazily inside one `asyncio.TaskGroup`; a semaphore of `pending_limit` slots (default `4 * max_inflight`), released by each task's done callback, bounds live tasks, and a second semaphore bounds concurrent `generate` calls to `max_inflight`. The first failure cancels the other tasks and is re-raised unwrapped (`_first_error` searches nested groups past cancellations); a failing job iterator propagates the same way. Progress reports one cumulative plan across calls.
- `SchedulerMetrics`: `requested`, `executed`, `deduplicated`, `cache_hits`, `kv_cache_hits`, `kv_cache_reused_tokens`; the scheduler leaves the two key-value (KV) cache counters at 0 and the orchestrator fills them from the backend.
- `CoalitionJob`: coalition key, snapshot id, rendered snapshot, absence policy, packed mask, model id, `prefix_hash`.
- `build_coalition_key(...)`: SHA-256 over snapshot id, player ids, packed mask, absence policy, model id and generation parameters; `CoalitionDedupRegistry`; `dedup_enabled(config, generation)` (`auto` is on at any temperature).
- `HotResultStore(maxsize)`: LRU map from coalition key to `StoredGeneration` (text, logprobs, `forced`, `audio_path`; `put`, `get`, `logprobs`, `entry`); the orchestrator sizes it to `2^n`.
- `RunArchive`: `open(root, manifest, flush_every, on_exists)` (raises `ArchiveExistsError` for a finished archive under `on_exists="error"`, always replaces an unfinished one), `load(root)` (opens the database read-only (`mode=ro`, no schema setup), so it writes nothing and works on read-only storage; `ArchiveNotFoundError`, a `FileNotFoundError`, without `manifest.json` or `archive.sqlite`; `ArchiveCorruptError` when the manifest is not a JSON object or the database cannot be read; `ManifestVersionError` on another schema), `write_manifest(manifest)`, `write_base_generation`, `append(draft)`, `mark_complete`, `is_complete`, `history_lazy()`. One SQLite file with WAL journaling, generation text, logprobs, teacher-forced scores (JSON, `NULL` without them) and audio path in the row, `ARCHIVE_SCHEMA_VERSION` (SQLite `user_version`) and `MANIFEST_SCHEMA_VERSION` 5, plus `manifest.json` and `base_generation.txt`.
- `read_manifest_payload(path)`: the manifest's JSON object, or `ArchiveCorruptError`; `check_manifest_schema_version(payload)`.
- `PrefixCacheManager(max_entries=8)`: `longest_prefix(token_ids)`, `store`, `reset`; LRU-bounded token-prefix states, croppable states (`CroppableCache`) matched on any shared leading tokens. `build_snapshot_prefix_hash(snapshot, player_set, mask)` hashes the presence of every player but the last ([[Prefix hash]]); `group_jobs_for_prefix_cache` sorts by it.
- `ObservabilitySink`, `NullObservabilitySink`, `InMemoryObservabilitySink`, `SpanRecord`; `CoalitionProgress`, `NullCoalitionProgress`; `PerfSummary`.

## Data flow

- In: a lazy `CoalitionJob` iterator and an async `generate(snapshot) -> str | GenerationRecord` from the [[pipeline]] orchestrator.
- Out: text and logprobs per coalition key in the store, archive rows (cache hits carry the stored logprobs), `SchedulerMetrics`.
- Imports [[domain]] (settings in `domain/config.py`) and [[masking]] (codec); no longer imports the pipeline, so the import cycle is gone.

## Invariants

- A coalition key identifies one generation request; duplicate keys reuse the stored text.
- A run either finishes (`mark_complete`) or leaves an archive `reanalyze` refuses; no partial success is reported.
- No "Task exception was never retrieved" warnings: every task belongs to the task group.
- `RunArchive.load` never writes: reading an archive leaves its files byte-identical.

## Known issues

- None open. Fixed: C02, C06, C07 ([[Fixed issues]]).

## Tests

- [[tests/runtime/test_scheduler.py]]
- [[tests/runtime/test_dedup.py]]
- [[tests/runtime/test_store.py]]
- [[tests/runtime/test_archive.py]]
- [[tests/runtime/test_kv_cache.py]]
- [[tests/runtime/test_perf_metrics.py]]
- [[tests/runtime/test_progress.py]]
- [[tests/runtime/test_telemetry.py]]
- [[tests/benchmarks/test_scheduler_bench.py]]
- [[tests/benchmarks/test_runtime_archive_bench.py]]
