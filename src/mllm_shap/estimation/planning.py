"""Budget planning from a target error (Maleki et al. 2013, Hoeffding bound).

A player's value is a mean of marginal contributions that lie in a range of width
``r``; sampling ``m`` random orderings keeps every player within ``epsilon`` of the
Shapley value with probability ``1 - beta`` once ``m >= r^2 ln(2 n / beta) / (2
epsilon^2)``, a union bound over the ``n`` players. The bound is worst-case and loose:
a pilot run gives a tighter range than the spread of the value function.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_planning.py]]
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetPlan:
    """Budget that meets a target error, in orderings and in evaluations."""

    num_players: int
    """Number of players of the game."""

    orderings: int
    """Random orderings the Hoeffding bound asks for."""

    evaluations: int
    """Coalitions the orderings can touch, ``orderings * (n - 1) + 2``, capped at the
    ``2^n`` of the full table."""

    exact_is_cheaper: bool
    """Whether enumerating all ``2^n`` coalitions costs no more than sampling."""

    @property
    def per_player(self) -> float:
        """Return :attr:`evaluations` per player, the unit of ``budget.per_player``."""
        return self.evaluations / self.num_players


def plan_budget(
    num_players: int,
    value_range: float,
    target_error: float,
    confidence: float = 0.95,
) -> BudgetPlan:
    """Return the orderings and evaluations for ``target_error`` at ``confidence``.

    ``value_range`` is the width of the marginal contributions, for example
    ``v(N) - v(empty)`` for a monotone game or the spread seen in a pilot run.

    Raises:
        ValueError: If a count is below 1 or an error, range or confidence is not
            positive or not below one for the confidence.
    """
    if num_players < 1:
        msg = "num_players must be at least 1"
        raise ValueError(msg)
    if value_range <= 0.0 or target_error <= 0.0:
        msg = "value_range and target_error must be positive"
        raise ValueError(msg)
    if not 0.0 < confidence < 1.0:
        msg = "confidence must lie strictly between 0 and 1"
        raise ValueError(msg)
    failure = (1.0 - confidence) / num_players
    orderings = math.ceil(
        value_range**2 * math.log(2.0 / failure) / (2.0 * target_error**2)
    )
    full_table = 1 << num_players
    evaluations = min(orderings * max(num_players - 1, 0) + 2, full_table)
    return BudgetPlan(
        num_players=num_players,
        orderings=orderings,
        evaluations=evaluations,
        exact_is_cheaper=evaluations >= full_table,
    )
