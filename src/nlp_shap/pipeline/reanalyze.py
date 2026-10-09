"""Reanalyze archived coalition generations without backend calls.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_reanalyze.py]]
"""

import dataclasses
import logging
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from ..domain.coalition import CoalitionMask
from ..errors import (
    ArchiveCorruptError,
    ArchiveIncompleteError,
    ReanalyzeMismatchError,
)
from ..estimation._shared import budget_fraction
from ..masking.codec import MaskCodec
from ..plugins.context import PluginContext
from ..plugins.groups import PluginGroup
from ..plugins.options import check_plugin_options
from ..plugins.registry import PluginRegistry
from ..protocols.estimand import EstimandAggregator
from ..protocols.estimator import EstimatorStrategy
from ..protocols.normalizer import Normalizer
from ..protocols.value import (
    RequiresEmbeddings,
    RequiresLogprobs,
    ValueFunction,
)
from ..runtime.archive import CoalitionRecord, RunArchive, read_manifest_payload
from ..runtime.metrics import PerfSummary
from ..runtime.scheduler import SchedulerMetrics
from ..runtime.telemetry import NullObservabilitySink, ObservabilitySink
from ..value.generation_records import generation_record_from_text
from .config import ExplainConfig
from .manifest import EmbeddingSource, RunManifest, parse_manifest
from .orchestrator import score_coalitions
from .result import ExplainResult, ExplainRunOutput

logger = logging.getLogger(__name__)


def reanalyze_sync(
    archive_root: Path,
    config: ExplainConfig,
    registry: PluginRegistry,
    *,
    manifest: RunManifest | None = None,
    telemetry: ObservabilitySink | None = None,
    strict: bool = False,
) -> ExplainRunOutput:
    """Rescore and re-aggregate an archived explain run without backend calls.

    The archived estimator re-aggregates the coalitions, so ``reanalyze``
    reproduces ``explain`` for the same value function. Archives store text
    only, so embedding value functions score hash embeddings of that text; when
    the archived run scored backend embeddings this logs a WARNING, or raises
    with ``strict``.

    Raises:
        ValueError: If the archive has no coalition rows or no base generation.
        ArchiveNotFoundError: If the manifest or the SQLite database is missing.
        ArchiveCorruptError: If the manifest or the database cannot be read.
        ManifestVersionError: If the archive was written by another release.
        ArchiveIncompleteError: If the archive was not written by
            :class:`~nlp_shap.pipeline.runner.ExplainRunner`, its run did not
            finish, it lacks the grand or empty coalition, or a full-budget run
            (``budget.fraction`` 1.0, which includes every ``exact`` run) lacks
            some of the 2^n coalitions.
        ReanalyzeMismatchError: If ``config`` uses a different estimator, absence
            policy or partition than the archived run, the value function needs
            logprobs and a row has none, or, with ``strict``, the value function
            reads embeddings and the archived run scored backend embeddings.
        ConfigError: If ``explanation.plugin_options`` names a plugin outside the
            run or holds an unknown option.
    """
    sink = telemetry or NullObservabilitySink()
    started = time.perf_counter()

    with sink.span("reanalyze.load"):
        archive = RunArchive.load(archive_root)
        try:
            resolved_manifest = manifest or parse_manifest(
                read_manifest_payload(archive_root / "manifest.json")
            )
            records = list(archive.history_lazy())
            base_text = archive.read_base_generation()
            complete = archive.is_complete
        except sqlite3.DatabaseError as exc:
            msg = f"cannot read archive database {archive_root / 'archive.sqlite'}: "
            msg += str(exc)
            raise ArchiveCorruptError(msg) from exc
        finally:
            archive.close()

    archived_config = resolved_manifest.config
    if archived_config is None or not resolved_manifest.estimator:
        msg = (
            f"archive {archive_root} was not written by ExplainRunner "
            "(its manifest records no config or estimator)"
        )
        raise ArchiveIncompleteError(msg)
    if not complete:
        msg = (
            f"archive {archive_root} belongs to a run that did not finish; "
            "re-run explain to rebuild it"
        )
        raise ArchiveIncompleteError(msg)
    if not records:
        msg = f"archive contains no coalition records: {archive_root}"
        raise ValueError(msg)
    if base_text is None:
        msg = f"archive missing base generation: {archive_root / 'base_generation.txt'}"
        raise ValueError(msg)

    explanation = config.explanation
    for field_name, current, archived in (
        ("estimator", explanation.estimator, resolved_manifest.estimator),
        (
            "absence_policy",
            explanation.absence_policy,
            resolved_manifest.absence_policy,
        ),
        ("players", explanation.players, resolved_manifest.players),
    ):
        if current != archived:
            msg = f"{field_name} {current!r} differs from archived {archived!r}"
            raise ReanalyzeMismatchError(msg)
    check_plugin_options(config, registry)
    plugin_context = PluginContext(config=config, registry=registry)
    estimator = cast(
        EstimatorStrategy,
        registry.resolve(
            PluginGroup.ESTIMATORS, resolved_manifest.estimator, plugin_context
        ),
    )
    # A coalition may be archived more than once (cache hits); score it once.
    seen: set[str] = set()
    unique_records = []
    for record in records:
        if record.coalition_key in seen:
            continue
        seen.add(record.coalition_key)
        unique_records.append(record)
    records = unique_records
    aggregator = cast(
        EstimandAggregator,
        registry.resolve(
            PluginGroup.ESTIMANDS, explanation.estimand.value, plugin_context
        ),
    )
    value_fn = cast(
        ValueFunction,
        registry.resolve(PluginGroup.VALUE_FNS, explanation.value_fn, plugin_context),
    )
    if (
        resolved_manifest.embedding_source is EmbeddingSource.BACKEND
        and isinstance(value_fn, RequiresEmbeddings)
        and value_fn.needs_embeddings
    ):
        msg = (
            f"archive {archive_root} was scored with embeddings of backend model "
            f"{resolved_manifest.embedding_model!r}; reanalyze scores hash "
            f"embeddings of the archived text, so {explanation.value_fn!r} values "
            "differ from explain"
        )
        if strict:
            raise ReanalyzeMismatchError(msg)
        logger.warning(msg)
    _check_logprobs(archive_root, value_fn, records, resolved_manifest)
    normalizer = cast(
        Normalizer,
        registry.resolve(
            PluginGroup.NORMALIZERS, explanation.normalizer, plugin_context
        ),
    )

    scoring_started = time.perf_counter()
    with sink.span("reanalyze.score"):
        base_record = generation_record_from_text(base_text)
        coalition_records = [
            dataclasses.replace(
                generation_record_from_text(record.generation_text),
                logprobs=record.logprobs,
            )
            for record in records
        ]
        payoffs = score_coalitions(value_fn, base_record, coalition_records)
    scoring_ms = (time.perf_counter() - scoring_started) * 1000.0

    aggregation_started = time.perf_counter()
    with sink.span("reanalyze.aggregate"):
        mask_presents = [list(MaskCodec.unpack(record.mask)) for record in records]
        lookup = dict(zip((tuple(m) for m in mask_presents), payoffs, strict=True))
        num_players = len(mask_presents[0])
        budget = archived_config.explanation.budget
        _check_coalition_rows(
            archive_root,
            budget_fraction(num_players, budget.fraction, budget.evaluations) == 1.0,
            lookup,
            num_players,
        )
        grand_value = lookup[(True,) * num_players]
        base_value = lookup[(False,) * num_players]
        raw_values = estimator.estimate_attributions(
            [CoalitionMask.from_sequence(present) for present in mask_presents],
            payoffs,
            aggregator,
        )
        normalized = normalizer.normalize(raw_values)
    aggregation_ms = (time.perf_counter() - aggregation_started) * 1000.0

    total_ms = (time.perf_counter() - started) * 1000.0
    metrics = SchedulerMetrics(
        requested=len(records),
        executed=0,
        deduplicated=0,
        cache_hits=len(records),
        kv_cache_hits=0,
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
        run_id=resolved_manifest.run_id,
        manifest=resolved_manifest,
        perf=PerfSummary(
            total_ms=total_ms,
            generation_ms=0.0,
            scoring_ms=scoring_ms,
            aggregation_ms=aggregation_ms,
        ),
    )


def _check_logprobs(
    archive_root: Path,
    value_fn: ValueFunction,
    records: Sequence[CoalitionRecord],
    manifest: RunManifest,
) -> None:
    """Refuse a value function that needs logprobs on rows stored without them."""
    if not (isinstance(value_fn, RequiresLogprobs) and value_fn.needs_logprobs):
        return
    if any(record.logprobs is None for record in records):
        msg = (
            f"archive {archive_root} has coalition rows without token logprobs "
            f"(backend {manifest.backend_kind!r}); value_fn {value_fn.name!r} "
            "cannot rescore it"
        )
        raise ReanalyzeMismatchError(msg)


def _check_coalition_rows(
    archive_root: Path,
    full_table: bool,
    lookup: dict[tuple[bool, ...], float],
    num_players: int,
) -> None:
    for name, present in (
        ("grand", (True,) * num_players),
        ("empty", (False,) * num_players),
    ):
        if present not in lookup:
            msg = f"archive {archive_root} has no {name} coalition row"
            raise ArchiveIncompleteError(msg)
    expected = 1 << num_players
    if full_table and len(lookup) != expected:
        msg = (
            f"archive {archive_root} holds {len(lookup)} of the {expected} "
            "coalitions of a full-budget run"
        )
        raise ArchiveIncompleteError(msg)
