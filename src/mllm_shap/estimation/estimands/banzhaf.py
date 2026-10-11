"""Banzhaf estimand aggregation.

@domain estimation
@see [[knowledge/architecture/estimation]]
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from ...domain.estimands import Estimand
from ...protocols.plugin import PluginOptions
from ._weights import aggregate_from_marginals, banzhaf_weight

if TYPE_CHECKING:
    from ...plugins.context import PluginContext


class BanzhafAggregator:
    """Aggregate coalition samples with uniform Banzhaf coalition weights."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def estimand(self) -> Estimand:
        """Return the Banzhaf estimand label."""
        return Estimand.BANZHAF

    def coalition_weight(self, coalition_size: int, num_players: int) -> float:
        """Return uniform Banzhaf coalition weight 1/2^(n-1)."""
        return banzhaf_weight(coalition_size, num_players)

    def aggregate(
        self,
        masks: Sequence[Sequence[bool]],
        payoffs: Sequence[float],
    ) -> list[float]:
        """Aggregate coalition samples into Banzhaf indices."""
        return aggregate_from_marginals(masks, payoffs, weight_fn=banzhaf_weight)
