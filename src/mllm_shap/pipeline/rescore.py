"""Rescore an archive with teacher-forced value functions, without generating.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_teacher_forcing.py]]
"""

import asyncio
from pathlib import Path
from typing import cast

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.generation import ForcedScore
from ..errors import ConfigError, ReanalyzeMismatchError
from ..masking.builder import MaskBuilder
from ..masking.codec import MaskCodec
from ..plugins.context import PluginContext
from ..plugins.groups import PluginGroup
from ..plugins.registry import PluginRegistry
from ..protocols.backend import GenerativeBackend, SupportsTeacherForcing
from ..protocols.masking import AbsencePolicy
from ..protocols.partition import PlayerPartitioner
from ..protocols.value import RequiresTeacherForcing, ValueFunction
from ..runtime.archive import RunArchive, read_manifest_payload
from ..runtime.telemetry import ObservabilitySink
from ..value.generation_records import generation_record_from_text
from .config import ExplainConfig
from .manifest import parse_manifest
from .reanalyze import reanalyze_sync
from .result import ExplainRunOutput


async def rescore(
    archive_root: Path,
    config: ExplainConfig,
    registry: PluginRegistry,
    snapshot: ConversationSnapshot,
    backend: SupportsTeacherForcing | GenerativeBackend,
    telemetry: ObservabilitySink | None = None,
) -> ExplainRunOutput:
    """Score an archived run with ``config``'s value function using forward passes.

    The coalition prompts are rebuilt from the archived masks and ``snapshot``
    (the conversation the run explained), the continuations the value function
    scores are run through ``backend.score_continuations``, and the archive is
    re-aggregated as :func:`~mllm_shap.pipeline.reanalyze.reanalyze_sync` does.
    No ``generate`` call is made. Value functions that need no teacher forcing
    give the same result as ``reanalyze``.

    Raises:
        ConfigError: If the value function needs teacher forcing and ``backend``
            cannot do it.
        ReanalyzeMismatchError: If ``snapshot`` does not partition into the
            archived players.
    """
    plugin_context = PluginContext(config=config, registry=registry)
    value_fn = cast(
        ValueFunction,
        registry.resolve(
            PluginGroup.VALUE_FNS, config.explanation.value_fn, plugin_context
        ),
    )
    forced: dict[str, dict[str, ForcedScore]] = {}
    if isinstance(value_fn, RequiresTeacherForcing) and value_fn.needs_teacher_forcing:
        forced = await _force_archive(
            archive_root, config, registry, snapshot, backend, value_fn
        )
    return await asyncio.to_thread(
        reanalyze_sync,
        archive_root,
        config,
        registry,
        telemetry=telemetry,
        forced=forced,
    )


async def _force_archive(
    archive_root: Path,
    config: ExplainConfig,
    registry: PluginRegistry,
    snapshot: ConversationSnapshot,
    backend: SupportsTeacherForcing | GenerativeBackend,
    value_fn: RequiresTeacherForcing,
) -> dict[str, dict[str, ForcedScore]]:
    if not (
        isinstance(backend, SupportsTeacherForcing) and backend.forces_continuations
    ):
        msg = "rescore needs a backend that scores continuations (transformers)"
        raise ConfigError(msg)
    archive = RunArchive.load(archive_root)
    try:
        manifest = parse_manifest(read_manifest_payload(archive_root / "manifest.json"))
        rows = list(archive.history_lazy())
        base_text = archive.read_base_generation() or ""
    finally:
        archive.close()
    context = PluginContext(config=config, registry=registry)
    partitioner = cast(
        PlayerPartitioner,
        registry.resolve(PluginGroup.PARTITIONS, config.explanation.players, context),
    )
    player_set = partitioner.partition(snapshot)
    if player_set.player_ids != manifest.player_ids:
        msg = f"snapshot players differ from the players archived in {archive_root}"
        raise ReanalyzeMismatchError(msg)
    policy = cast(
        AbsencePolicy,
        registry.resolve(
            PluginGroup.ABSENCE_POLICIES, manifest.absence_policy, context
        ),
    )
    builder = MaskBuilder(policy)
    continuations = value_fn.forced_continuations(
        generation_record_from_text(base_text)
    )
    out: dict[str, dict[str, ForcedScore]] = {}
    for row in rows:
        if row.coalition_key in out:
            continue
        mask = CoalitionMask.from_sequence(tuple(MaskCodec.unpack(row.mask)))
        rendered = builder.render(builder.view(snapshot, player_set, mask))
        scores = await backend.score_continuations(
            rendered, continuations, value_fn.forced_top_k
        )
        out[row.coalition_key] = dict(zip(continuations, scores, strict=True))
    return out
