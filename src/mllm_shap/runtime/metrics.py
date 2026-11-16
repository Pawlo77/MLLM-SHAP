"""Performance summaries for explain pipeline runs.

@domain runtime
@see [[knowledge/architecture/runtime]]
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostCounters:
    """Model work: counters are cumulative on a backend and per run in a summary."""

    forward_sequences: int = 0
    """Prompts (and prompt plus continuation pairs) run through the model."""

    prompt_tokens: int = 0
    """Input tokens processed, including forced continuation tokens."""

    generated_tokens: int = 0
    """New tokens produced by generation."""

    device_seconds: float = 0.0
    """Wall-clock seconds inside model calls, after device synchronisation."""

    def __sub__(self, other: "CostCounters") -> "CostCounters":
        """Return the work done since ``other`` was read."""
        return CostCounters(
            self.forward_sequences - other.forward_sequences,
            self.prompt_tokens - other.prompt_tokens,
            self.generated_tokens - other.generated_tokens,
            self.device_seconds - other.device_seconds,
        )


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

    forward_sequences: int = 0
    """Model sequences run during the run; the four cost fields are zero when the
    backend reports none, and for ``reanalyze``, which answers from the archive."""

    prompt_tokens: int = 0
    """Input tokens processed during the run."""

    generated_tokens: int = 0
    """New tokens generated during the run."""

    device_seconds: float = 0.0
    """Seconds inside model calls during the run."""
