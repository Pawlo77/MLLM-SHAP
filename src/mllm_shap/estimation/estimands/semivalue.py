"""Semivalue estimands: Beta Shapley and weighted Banzhaf.

Both give each coalition size a probability and read it off the same sampled
stratum means as Shapley and Banzhaf, so any estimator that samples across sizes
(``mc``, ``exact``) serves them.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/estimands/test_semivalue.py]]
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field, field_validator

from ...domain.estimands import Estimand
from ...protocols.plugin import PluginOptions
from ._weights import aggregate_from_marginals

if TYPE_CHECKING:
    from ...plugins.context import PluginContext


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_shapley_weight(
    coalition_size: int, num_players: int, alpha: float, beta: float
) -> float:
    """Return the Beta(``alpha``, ``beta``) Shapley weight of one coalition.

    Following Kwon and Zou (2022), the inclusion probability is Beta(``beta``,
    ``alpha``) distributed, so ``alpha > beta`` favours small coalitions and
    ``alpha = beta = 1`` is Shapley.
    """
    return math.exp(
        _log_beta(coalition_size + beta, num_players - 1 - coalition_size + alpha)
        - _log_beta(alpha, beta)
    )


def weighted_banzhaf_weight(coalition_size: int, num_players: int, p: float) -> float:
    """Return ``p^s (1 - p)^(n - 1 - s)``; ``p = 1/2`` is Banzhaf."""
    return p**coalition_size * (1.0 - p) ** (num_players - 1 - coalition_size)


class BetaShapleyOptions(PluginOptions):
    """Beta Shapley shape parameters."""

    alpha: float = Field(default=4.0, gt=0.0)
    """Mass on small coalitions; with ``beta`` it sets Beta(alpha, beta)."""

    beta: float = Field(default=1.0, gt=0.0)
    """Second shape parameter; ``alpha = beta = 1`` gives Shapley."""


class WeightedBanzhafOptions(PluginOptions):
    """Weighted Banzhaf inclusion probability."""

    p: float = Field(default=0.3, gt=0.0)
    """Probability that each other player is present in a coalition; below 1."""

    @field_validator("p")
    @classmethod
    def _below_one(cls, value: float) -> float:
        if value >= 1.0:
            msg = "p must be below 1"
            raise ValueError(msg)
        return value


class BetaShapleyAggregator:
    """Aggregate coalition samples with Beta Shapley weights."""

    options_model: ClassVar[type[PluginOptions]] = BetaShapleyOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read ``alpha`` and ``beta`` from the plugin options in ``context``."""
        options = BetaShapleyOptions.model_validate(context.options if context else {})
        self.alpha = options.alpha
        self.beta = options.beta

    @property
    def estimand(self) -> Estimand:
        """Return the Beta Shapley estimand label."""
        return Estimand.BETA_SHAPLEY

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return the Beta Shapley weight of a coalition without the player."""
        return beta_shapley_weight(coalition_size, num_players, self.alpha, self.beta)

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into Beta Shapley values."""
        return aggregate_from_marginals(masks, payoffs, weight_fn=self.coalition_weight)


class WeightedBanzhafAggregator:
    """Aggregate coalition samples with weighted Banzhaf weights."""

    options_model: ClassVar[type[PluginOptions]] = WeightedBanzhafOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read ``p`` from the plugin options in ``context``."""
        options = WeightedBanzhafOptions.model_validate(
            context.options if context else {}
        )
        self.p = options.p

    @property
    def estimand(self) -> Estimand:
        """Return the weighted Banzhaf estimand label."""
        return Estimand.WEIGHTED_BANZHAF

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return the weighted Banzhaf weight of a coalition without the player."""
        return weighted_banzhaf_weight(coalition_size, num_players, self.p)

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into weighted Banzhaf indices."""
        return aggregate_from_marginals(masks, payoffs, weight_fn=self.coalition_weight)
