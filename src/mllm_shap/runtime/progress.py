"""Optional synchronous progress callbacks for coalition execution.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_progress.py]]
"""

from typing import Protocol


class CoalitionProgress(Protocol):
    """Sync hooks notified as coalition jobs are planned and finished."""

    def on_coalitions_planned(self, total: int) -> None:
        """Report the number of coalition jobs planned so far in the run.

        Called again with a larger ``total`` when a second sampling phase
        (Neyman) adds jobs; finished counts keep growing across phases.
        """

    def on_coalition_finished(self, done: int, total: int) -> None:
        """Report that ``done`` of ``total`` coalition jobs have finished."""


class NullCoalitionProgress:
    """No-op progress sink used when callers omit a progress callback."""

    def on_coalitions_planned(self, total: int) -> None:
        """Ignore planned coalition count."""
        _ = total

    def on_coalition_finished(self, done: int, total: int) -> None:
        """Ignore finished coalition count."""
        _ = done
        _ = total
