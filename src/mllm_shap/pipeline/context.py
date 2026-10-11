"""Shared state for one explain pipeline run.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
"""

from dataclasses import dataclass
from pathlib import Path

from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..plugins.context import PluginContext
from ..plugins.registry import PluginRegistry
from ..runtime.progress import CoalitionProgress, NullCoalitionProgress
from ..runtime.telemetry import NullObservabilitySink, ObservabilitySink
from .config import ExplainConfig


@dataclass(frozen=True, slots=True)
class ExplainContext:
    """Inputs resolved for one explain orchestrator run."""

    config: ExplainConfig
    """Validated explain configuration."""

    snapshot: ConversationSnapshot
    """Conversation under explanation."""

    player_set: PlayerSet
    """Explainability players derived from the snapshot."""

    registry: PluginRegistry
    """Plugin registry used to resolve pipeline components."""

    run_id: str
    """Stable identifier for archive paths and manifests."""

    archive_root: Path | None = None
    """Optional filesystem root for run archive persistence."""

    telemetry: ObservabilitySink | None = None
    """Optional observability sink for stage spans."""

    progress: CoalitionProgress | None = None
    """Optional sync sink for coalition scheduling progress."""

    def plugin_context(self) -> PluginContext:
        """Return the context passed to plugin factories for this run."""
        return PluginContext(config=self.config, registry=self.registry)

    def resolved_telemetry(self) -> ObservabilitySink:
        """Return the configured telemetry sink or a no-op implementation."""
        return self.telemetry or NullObservabilitySink()

    def resolved_progress(self) -> CoalitionProgress:
        """Return the configured progress sink or a no-op implementation."""
        return self.progress or NullCoalitionProgress()
