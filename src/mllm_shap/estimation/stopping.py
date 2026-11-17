"""Stopping rule: end sampling once the estimate is narrow or its top set is stable.

The budget stays the upper bound. After every ``check_every`` new coalitions, past a
floor, sampling ends when every player's 95% interval is at most ``max_ci_width`` wide
(needs standard errors) or the ``top_k_stable`` players of largest magnitude are the
same set as at the previous check.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_stopping.py]]
"""

from collections.abc import Sequence
from dataclasses import dataclass

Z_95 = 1.959964
"""Normal quantile of a two-sided 95% interval."""


@dataclass(frozen=True, slots=True)
class StopDecision:
    """Outcome of one stopping check."""

    stop: bool
    """Whether sampling should end now."""

    reason: str | None
    """``"ci_width"`` or ``"top_k_stable"`` when :attr:`stop` is true."""

    top: tuple[int, ...] | None
    """The current top-``k`` players, to compare with at the next check."""


def top_players(values: Sequence[float], k: int) -> tuple[int, ...]:
    """Return the ``k`` players of largest magnitude, sorted by index."""
    ranked = sorted(range(len(values)), key=lambda i: (-abs(values[i]), i))
    return tuple(sorted(ranked[:k]))


def check_stop(
    values: Sequence[float],
    errors: Sequence[float] | None,
    max_ci_width: float | None,
    top_k_stable: int | None,
    previous_top: tuple[int, ...] | None,
) -> StopDecision:
    """Decide whether to stop given the current values and standard errors."""
    top = None if top_k_stable is None else top_players(values, top_k_stable)
    if max_ci_width is not None and errors is not None:
        widest = 2.0 * Z_95 * max(errors, default=0.0)
        if widest <= max_ci_width:
            return StopDecision(True, "ci_width", top)
    if top is not None and top == previous_top:
        return StopDecision(True, "top_k_stable", top)
    return StopDecision(False, None, top)
