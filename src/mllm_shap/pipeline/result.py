"""Explain pipeline result types.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_result.py]]
"""

from dataclasses import asdict, dataclass
from typing import TypedDict

from ..domain.estimands import Estimand, EstimandWire, estimand_to_wire
from ..domain.players import PlayerSet
from ..runtime.metrics import PerfSummary
from ..runtime.scheduler import SchedulerMetrics
from .manifest import RunManifest, RunManifestPayload


class PlayerLabelPayload(TypedDict):
    """JSON form of one player: its id and, when known, where it sits."""

    player_id: str
    """Player identifier in mask bit order."""

    text: str | None
    """Text the player covers; ``None`` when the output has no player spans."""

    message_index: int | None
    """Message index (turns flattened); ``None`` without player spans."""

    start: int | None
    """Start character offset in the message; ``None`` without player spans."""

    end: int | None
    """End character offset (exclusive); ``None`` without player spans."""

    speaker: str | None
    """Speaker of the player's message; ``None`` when unset or unknown."""

    label: str | None
    """Display label (``PlayerSpan.display_label``); ``None`` without spans."""


class ExplainRunPayload(TypedDict):
    """JSON form of an explain or reanalyze run, keys in a stable order."""

    values: list[float]
    """Per-player attribution values."""

    labels: list[PlayerLabelPayload]
    """One entry per player, aligned with ``values``."""

    base_value: float
    """Payoff of the empty coalition v(empty)."""

    grand_value: float
    """Payoff of the grand coalition v(N)."""

    estimand: EstimandWire
    """Estimand used to aggregate the coalitions."""

    metrics: dict[str, int]
    """Scheduler counters (:class:`~mllm_shap.runtime.scheduler.SchedulerMetrics`)."""

    manifest: RunManifestPayload
    """Run manifest, as written next to the archive."""

    perf: dict[str, float] | None
    """Wall-clock breakdown in milliseconds, when measured."""


@dataclass(frozen=True, slots=True)
class ExplainResult:
    """Attribution output for a single explain run."""

    estimand: Estimand
    """Estimand used to aggregate coalition samples."""

    values: tuple[float, ...]
    """Per-player attribution values."""

    base_value: float = 0.0
    """Payoff of the empty coalition v(empty); Shapley values sum to v(N) - v(empty)."""

    grand_value: float = 0.0
    """Payoff of the grand coalition v(N), scored against the base generation."""


@dataclass(frozen=True, slots=True)
class ExplainRunOutput:
    """Explain pipeline output including scheduler metrics."""

    result: ExplainResult
    """Aggregated and normalized attribution values."""

    metrics: SchedulerMetrics
    """Coalition scheduler counters for the run."""

    run_id: str
    """Stable identifier for the explain run."""

    manifest: RunManifest
    """Manifest metadata persisted with optional archives."""

    perf: PerfSummary | None = None
    """Optional wall-clock breakdown for the run."""

    player_set: PlayerSet | None = None
    """Players of the explained snapshot; ``None`` for reanalyzed runs."""

    def to_dict(self) -> ExplainRunPayload:
        """Return a JSON-compatible payload with a stable key order.

        Labels carry each player's text and span when :attr:`player_set` is
        known (``explain``); after ``reanalyze`` only the archived player ids
        are available and the span fields are ``None``.
        """
        return {
            "values": list(self.result.values),
            "labels": self._labels(),
            "base_value": self.result.base_value,
            "grand_value": self.result.grand_value,
            "estimand": estimand_to_wire(self.result.estimand),
            "metrics": asdict(self.metrics),
            "manifest": self.manifest.to_dict(),
            "perf": None if self.perf is None else asdict(self.perf),
        }

    def _labels(self) -> list[PlayerLabelPayload]:
        if self.player_set is not None and self.player_set.spans:
            return [
                {
                    "player_id": player_id,
                    "text": span.text,
                    "message_index": span.message_index,
                    "start": span.start,
                    "end": span.end,
                    "speaker": span.speaker,
                    "label": span.display_label,
                }
                for player_id, span in zip(
                    self.player_set.player_ids, self.player_set.spans, strict=True
                )
            ]
        return [
            {
                "player_id": player_id,
                "text": None,
                "message_index": None,
                "start": None,
                "end": None,
                "speaker": None,
                "label": None,
            }
            for player_id in self.manifest.player_ids
        ]
