"""Explain pipeline orchestration.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_orchestrator.py]]
"""

import dataclasses
import logging
import time
from collections.abc import Iterator, Sequence
from typing import cast

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.enums import EmbeddingMode
from ..domain.generation import GenerationRecord
from ..domain.players import PlayerSet
from ..errors import ConfigError
from ..estimation._shared import budget_fraction
from ..estimation.normalizers import apply_efficiency
from ..masking.builder import MaskBuilder
from ..masking.codec import MaskCodec
from ..plugins.groups import PluginGroup
from ..protocols.backend import (
    GenerationResult,
    GenerativeBackend,
    SupportsKvCache,
    SupportsLogprobs,
)
from ..protocols.estimand import EstimandAggregator
from ..protocols.estimator import EstimatorStrategy, TwoPhaseEstimator
from ..protocols.masking import AbsencePolicy
from ..protocols.normalizer import Normalizer
from ..protocols.value import (
    FittableValueFunction,
    RequiresEmbeddings,
    RequiresLogprobs,
    ValueFunction,
)
from ..runtime.archive import BASE_GENERATION_FILE, RunArchive
from ..runtime.dedup import CoalitionDedupRegistry, build_coalition_key, dedup_enabled
from ..runtime.kv_cache import build_snapshot_prefix_hash
from ..runtime.metrics import PerfSummary
from ..runtime.scheduler import (
    CoalitionJob,
    GenerateFn,
    InferenceScheduler,
    SchedulerMetrics,
)
from ..runtime.store import HotResultStore
from ..value.generation_records import generation_record_from_text
from .context import ExplainContext
from .manifest import EmbeddingSource, RunManifest
from .result import ExplainResult, ExplainRunOutput

logger = logging.getLogger(__name__)


class ExplainOrchestrator:
    """Wire partition, sampling, generation, scoring, and aggregation."""

    def __init__(
        self,
        context: ExplainContext,
        backend: GenerativeBackend,
    ) -> None:
        """Bind the run context and the backend that generates coalitions."""
        self._context = context
        self._backend = backend
        self._generation = context.config.generation
        self._kv_cache_baseline = (0, 0)
        # Backend records with model embeddings, by generated text; filled only
        # when the value function needs embeddings.
        self._embedded: dict[str, GenerationRecord] = {}

    async def run(self) -> ExplainRunOutput:
        """Execute the configured explain pipeline and return attributions."""
        telemetry = self._context.resolved_telemetry()
        run_started = time.perf_counter()
        generation_ms = 0.0
        scoring_ms = 0.0
        aggregation_ms = 0.0

        with telemetry.span("orchestrator.run"):
            config = self._context.config
            explanation = config.explanation
            registry = self._context.registry
            plugin_context = self._context.plugin_context()
            snapshot = self._context.snapshot
            player_set = self._context.player_set
            logger.info(
                "explain run %s started: %d players, estimator %s, backend %s",
                self._context.run_id,
                player_set.num_players,
                explanation.estimator,
                config.backend.kind,
            )

            estimator = cast(
                EstimatorStrategy,
                registry.resolve(
                    PluginGroup.ESTIMATORS, explanation.estimator, plugin_context
                ),
            )
            estimator.bind_snapshot(snapshot)

            aggregator = cast(
                EstimandAggregator,
                registry.resolve(
                    PluginGroup.ESTIMANDS,
                    explanation.estimand.value,
                    plugin_context,
                ),
            )

            value_fn = cast(
                ValueFunction,
                registry.resolve(
                    PluginGroup.VALUE_FNS, explanation.value_fn, plugin_context
                ),
            )

            self._check_logprobs(value_fn)
            if (
                isinstance(value_fn, RequiresEmbeddings)
                and value_fn.needs_embeddings
                and not self._generation.compute_embeddings
            ):
                self._generation = self._generation.model_copy(
                    update={"compute_embeddings": True}
                )

            normalizer = cast(
                Normalizer,
                registry.resolve(
                    PluginGroup.NORMALIZERS, explanation.normalizer, plugin_context
                ),
            )

            absence_policy = cast(
                AbsencePolicy,
                registry.resolve(
                    PluginGroup.ABSENCE_POLICIES,
                    explanation.absence_policy,
                    plugin_context,
                ),
            )
            mask_builder = MaskBuilder(absence_policy)

            archive: RunArchive | None = None
            manifest = RunManifest(
                estimand=explanation.estimand,
                run_id=self._context.run_id,
                snapshot_id=snapshot.snapshot_id,
                estimator=explanation.estimator,
                value_fn=explanation.value_fn,
                normalizer=explanation.normalizer,
                absence_policy=explanation.absence_policy,
                players=explanation.players,
                backend_kind=config.backend.kind,
                model_id=self._backend.model_id,
                player_ids=player_set.player_ids,
                config=config,
            )
            if self._context.archive_root is not None:
                archive = RunArchive.open(
                    self._context.archive_root,
                    manifest,
                    flush_every=explanation.archive.flush_every,
                    on_exists=explanation.archive.on_exists,
                )

            try:
                base_started = time.perf_counter()
                with telemetry.span("orchestrator.generate_base"):
                    base_record = await self._generate_base(mask_builder, player_set)
                generation_ms += (time.perf_counter() - base_started) * 1000.0
                logger.info("base generated in %.1f ms", generation_ms)

                if archive is not None:
                    archive.write_base_generation(base_record.text)
                    archive_root = self._context.archive_root
                    if archive_root is None:
                        msg = "archive root missing while archive is open"
                        raise RuntimeError(msg)
                    base_path = archive_root / BASE_GENERATION_FILE
                    if not base_path.is_file():
                        msg = "base generation was not persisted before coalition loop"
                        raise RuntimeError(msg)

                with telemetry.span("estimator.sample_masks"):
                    masks = self._ensure_boundary_masks(
                        list(
                            estimator.sample_masks(
                                player_set,
                                budget_fraction(
                                    player_set.num_players,
                                    explanation.budget.fraction,
                                    explanation.budget.evaluations_for(
                                        player_set.num_players
                                    ),
                                ),
                                explanation.include_minimal_masks,
                                explanation.seed,
                            )
                        ),
                        player_set.num_players,
                    )

                grand_mask = CoalitionMask.from_sequence(
                    (True,) * player_set.num_players
                )
                empty_mask = CoalitionMask.from_sequence(
                    (False,) * player_set.num_players
                )
                # Store is the scoring buffer for this run and must retain every
                # coalition output, including phase-two masks of two-phase
                # estimators (default LRU is 128).
                store = HotResultStore(maxsize=1 << player_set.num_players)
                # The base generation is v(N): seed it so the grand job is a cache hit.
                store.put(
                    self._coalition_key(mask_builder, player_set, grand_mask),
                    base_record.text,
                    base_record.logprobs,
                )
                dedup = (
                    CoalitionDedupRegistry()
                    if dedup_enabled(explanation.dedup, config.generation)
                    else None
                )
                scheduler = InferenceScheduler(
                    max_inflight=explanation.max_inflight,
                    generation=config.generation,
                    store=store,
                    dedup=dedup,
                    archive=archive,
                    progress=self._context.resolved_progress(),
                )

                if isinstance(self._backend, SupportsKvCache):
                    self._backend.set_kv_cache_enabled(
                        explanation.kv_cache.enabled, explanation.kv_cache.max_entries
                    )
                    # Backend counters are cumulative; metrics report this run only.
                    self._kv_cache_baseline = (
                        self._backend.kv_cache_hits,
                        self._backend.kv_cache_reused_tokens,
                    )

                async def generate(
                    snapshot_for_job: ConversationSnapshot,
                ) -> str | GenerationRecord:
                    record = await self._backend.generate(
                        snapshot_for_job, self._generation
                    )
                    self._keep_embeddings(record)
                    return (
                        record if isinstance(record, GenerationRecord) else record.text
                    )

                schedule_started = time.perf_counter()
                coalition_records, metrics = await self._generate_records(
                    mask_builder, player_set, masks, scheduler, store, generate
                )
                generation_ms += (time.perf_counter() - schedule_started) * 1000.0

                scoring_started = time.perf_counter()
                payoffs = self._score(value_fn, base_record, coalition_records)
                scoring_ms += (time.perf_counter() - scoring_started) * 1000.0

                if isinstance(estimator, TwoPhaseEstimator):
                    estimator.begin_allocation(masks, payoffs)
                    phase_two = list(estimator.sample_allocation_masks())
                    if phase_two:
                        schedule_started = time.perf_counter()
                        extra_records, extra_metrics = await self._generate_records(
                            mask_builder,
                            player_set,
                            phase_two,
                            scheduler,
                            store,
                            generate,
                        )
                        generation_ms += (
                            time.perf_counter() - schedule_started
                        ) * 1000.0
                        masks.extend(phase_two)
                        coalition_records.extend(extra_records)
                        metrics = _merge_metrics(metrics, extra_metrics)
                        # Rescore every coalition so corpus-fitted value functions
                        # (TF-IDF) score both phases against the same corpus.
                        scoring_started = time.perf_counter()
                        payoffs = self._score(value_fn, base_record, coalition_records)
                        scoring_ms += (time.perf_counter() - scoring_started) * 1000.0

                manifest = self._with_embedding_source(manifest, value_fn)
                if archive is not None:
                    archive.write_manifest(manifest)
                    archive.mark_complete()

                payoff_by_mask = dict(
                    zip((m.present for m in masks), payoffs, strict=True)
                )
                grand_value = payoff_by_mask[grand_mask.present]
                base_value = payoff_by_mask[empty_mask.present]

                aggregation_started = time.perf_counter()
                with telemetry.span("orchestrator.aggregate"):
                    raw_values = estimator.estimate_attributions(
                        masks, payoffs, aggregator
                    )
                    raw_values = apply_efficiency(
                        explanation.efficiency, raw_values, grand_value, base_value
                    )
                    normalized = normalizer.normalize(raw_values)
                aggregation_ms = (time.perf_counter() - aggregation_started) * 1000.0
            finally:
                if archive is not None:
                    archive.close()

        total_ms = (time.perf_counter() - run_started) * 1000.0
        logger.info(
            "explain run %s done in %.1f ms (%d coalitions, %d backend calls)",
            self._context.run_id,
            total_ms,
            metrics.requested,
            metrics.executed,
        )
        result = ExplainResult(
            estimand=explanation.estimand,
            values=tuple(normalized),
            base_value=base_value,
            grand_value=grand_value,
        )
        return ExplainRunOutput(
            result=result,
            metrics=metrics,
            run_id=self._context.run_id,
            manifest=manifest,
            player_set=self._context.player_set,
            perf=PerfSummary(
                total_ms=total_ms,
                generation_ms=generation_ms,
                scoring_ms=scoring_ms,
                aggregation_ms=aggregation_ms,
            ),
        )

    async def _generate_base(
        self,
        mask_builder: MaskBuilder,
        player_set: PlayerSet,
    ) -> GenerationRecord:
        grand_mask = CoalitionMask.from_sequence((True,) * player_set.num_players)
        masked = mask_builder.view(
            self._context.snapshot,
            player_set,
            grand_mask,
        )
        rendered = mask_builder.render(masked)
        generated = await self._backend.generate(rendered, self._generation)
        self._keep_embeddings(generated)
        logprobs = (
            generated.logprobs if isinstance(generated, GenerationRecord) else None
        )
        return dataclasses.replace(
            self._scoring_record(generated.text), logprobs=logprobs
        )

    def _check_logprobs(self, value_fn: ValueFunction) -> None:
        """Refuse a value function that needs logprobs on a backend without them.

        Raises:
            ConfigError: Naming the backend, before any generation runs.
        """
        if not (isinstance(value_fn, RequiresLogprobs) and value_fn.needs_logprobs):
            return
        if isinstance(self._backend, SupportsLogprobs) and (
            self._backend.returns_logprobs
        ):
            return
        msg = (
            f"value_fn {self._context.config.explanation.value_fn!r} needs token "
            f"logprobs, which backend {self._context.config.backend.kind!r} does "
            "not return; use the transformers or api backend"
        )
        raise ConfigError(msg)

    def _with_embedding_source(
        self, manifest: RunManifest, value_fn: ValueFunction
    ) -> RunManifest:
        """Record in ``manifest`` which embeddings the scored records carried.

        Only the vector ``explanation.embedding_mode`` selects is compared:
        backend vectors equal to the text-derived hash vectors ``reanalyze``
        scores (the mock, LM Studio and API backends' vectors) count as
        ``"hash"``.
        """
        contextual = (
            self._context.config.explanation.embedding_mode is EmbeddingMode.CONTEXTUAL
        )
        for text in self._embedded:
            scored = self._scoring_record(text)
            hashed = generation_record_from_text(text)
            if (
                scored.contextual_embedding != hashed.contextual_embedding
                if contextual
                else scored.embedding != hashed.embedding
            ):
                return dataclasses.replace(
                    manifest,
                    embedding_source=EmbeddingSource.BACKEND,
                    embedding_model=self._backend.model_id,
                )
        if isinstance(value_fn, RequiresEmbeddings) and value_fn.needs_embeddings:
            return dataclasses.replace(manifest, embedding_source=EmbeddingSource.HASH)
        return manifest

    def _keep_embeddings(self, record: GenerationResult) -> None:
        if (
            self._generation.compute_embeddings
            and isinstance(record, GenerationRecord)
            and record.embedding
        ):
            self._embedded[record.text] = record

    def _scoring_record(self, text: str) -> GenerationRecord:
        """Return the record scored for ``text``.

        It is the text-derived record ``reanalyze`` also scores, with the
        backend's model embeddings swapped in when they were requested.
        """
        record = generation_record_from_text(text)
        backend_record = self._embedded.get(text)
        if backend_record is None:
            return record
        return dataclasses.replace(
            record,
            embedding=backend_record.embedding,
            contextual_embedding=(
                backend_record.contextual_embedding or record.contextual_embedding
            ),
        )

    def _iter_jobs(
        self,
        mask_builder: MaskBuilder,
        planned: Sequence[tuple[CoalitionMask, str, str]],
    ) -> Iterator[CoalitionJob]:
        """Render one job at a time from ``(mask, coalition_key, prefix_hash)``."""
        for mask, coalition_key, prefix_hash in planned:
            masked = mask_builder.view(
                self._context.snapshot, self._context.player_set, mask
            )
            packed = MaskCodec.pack(mask.present)
            yield CoalitionJob(
                coalition_key=coalition_key,
                snapshot_id=self._context.snapshot.snapshot_id,
                snapshot=mask_builder.render(masked),
                absence_policy=mask_builder.policy_name,
                mask_words=packed.words,
                mask_n_bits=packed.n_bits,
                prefix_hash=prefix_hash,
                model_id=self._backend.model_id,
            )

    @staticmethod
    def _ensure_boundary_masks(
        masks: list[CoalitionMask],
        num_players: int,
    ) -> list[CoalitionMask]:
        """Append the empty and grand coalitions when the sampler omitted them."""
        present = {mask.present for mask in masks}
        grand = (True,) * num_players
        empty = (False,) * num_players
        if grand not in present:
            masks.append(CoalitionMask.from_sequence(grand))
        if empty not in present:
            masks.append(CoalitionMask.from_sequence(empty))
        return masks

    def _coalition_key(
        self,
        mask_builder: MaskBuilder,
        player_set: PlayerSet,
        mask: CoalitionMask,
    ) -> str:
        return build_coalition_key(
            snapshot_id=self._context.snapshot.snapshot_id,
            player_ids=player_set.player_ids,
            mask_present=mask.present,
            absence_policy=mask_builder.policy_name,
            model_id=self._backend.model_id,
            generation=self._context.config.generation,
        )

    async def _generate_records(
        self,
        mask_builder: MaskBuilder,
        player_set: PlayerSet,
        masks: Sequence[CoalitionMask],
        scheduler: InferenceScheduler,
        store: HotResultStore,
        generate: GenerateFn,
    ) -> tuple[list[GenerationRecord], SchedulerMetrics]:
        """Schedule ``masks`` and return their records in mask order.

        Jobs are rendered lazily while the scheduler consumes them. With the KV
        cache enabled they run grouped by prompt prefix; records are read back
        from ``store`` by coalition key, so they stay aligned with ``masks``.
        """
        telemetry = self._context.resolved_telemetry()
        planned = [
            (
                mask,
                self._coalition_key(mask_builder, player_set, mask),
                build_snapshot_prefix_hash(self._context.snapshot, player_set, mask),
            )
            for mask in masks
        ]
        scheduled = (
            sorted(planned, key=lambda item: item[2])
            if self._context.config.explanation.kv_cache.enabled
            else planned
        )
        logger.info("%d coalitions scheduled", len(planned))
        with telemetry.span("orchestrator.schedule"):
            metrics = await scheduler.run_iter(
                self._iter_jobs(mask_builder, scheduled),
                generate,
                total=len(planned),
            )
        if isinstance(self._backend, SupportsKvCache):
            hits_before, tokens_before = self._kv_cache_baseline
            metrics = dataclasses.replace(
                metrics,
                kv_cache_hits=self._backend.kv_cache_hits - hits_before,
                kv_cache_reused_tokens=(
                    self._backend.kv_cache_reused_tokens - tokens_before
                ),
            )
        records: list[GenerationRecord] = []
        for _, coalition_key, _ in planned:
            text = store.get(coalition_key)
            if text is None:
                msg = f"missing generation for coalition {coalition_key!r}"
                raise RuntimeError(msg)
            records.append(
                dataclasses.replace(
                    self._scoring_record(text), logprobs=store.logprobs(coalition_key)
                )
            )
        return records, metrics

    def _score(
        self,
        value_fn: ValueFunction,
        base_record: GenerationRecord,
        coalition_records: list[GenerationRecord],
    ) -> list[float]:
        """Fit corpus-dependent value functions and score every coalition."""
        with self._context.resolved_telemetry().span("orchestrator.score"):
            return score_coalitions(value_fn, base_record, coalition_records)


def score_coalitions(
    value_fn: ValueFunction,
    base_record: GenerationRecord,
    coalition_records: Sequence[GenerationRecord],
) -> list[float]:
    """Fit a corpus-dependent ``value_fn`` on all records, then score each coalition."""
    if isinstance(value_fn, FittableValueFunction):
        value_fn.fit((base_record, *coalition_records))
    return [value_fn.score(base_record, record) for record in coalition_records]


def _merge_metrics(
    first: SchedulerMetrics,
    second: SchedulerMetrics,
) -> SchedulerMetrics:
    """Sum scheduler counters from two scheduling passes.

    ``kv_cache_hits`` and ``kv_cache_reused_tokens`` are counted from the start
    of the run, so the later pass already includes the earlier one.
    """
    return SchedulerMetrics(
        requested=first.requested + second.requested,
        executed=first.executed + second.executed,
        deduplicated=first.deduplicated + second.deduplicated,
        cache_hits=first.cache_hits + second.cache_hits,
        kv_cache_hits=max(first.kv_cache_hits, second.kv_cache_hits),
        kv_cache_reused_tokens=max(
            first.kv_cache_reused_tokens, second.kv_cache_reused_tokens
        ),
    )
