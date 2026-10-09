"""Performance summaries for explain pipeline runs.

@domain runtime
@see [[knowledge/architecture/runtime]]
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PerfSummary:
    """Wall-clock breakdown for one explain or reanalyze run."""

    total_ms: float
    """Total wall-clock time for the run."""

    generation_ms: float
    """Time spent in base and coalition generation."""

    scoring_ms: float
    """Time spent scoring coalition outputs with the value function."""

    aggregation_ms: float
    """Time spent aggregating coalition payoffs into attributions."""
