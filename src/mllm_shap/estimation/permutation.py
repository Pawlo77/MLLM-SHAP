"""Permutation sampling with antithetic orderings.

Random orderings of the players are drawn and every prefix of each ordering is
evaluated, so one ordering costs ``n - 1`` new coalitions. An antithetic ordering is
the reverse of its partner: its prefix of size ``k`` is the complement of the
partner's prefix of size ``n - k``, which cancels part of the sampling noise.
Prefixes feed the estimand aggregator, so any estimand works.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_permutation.py]]
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ..protocols.plugin import PluginOptions
from ._shared import (
    _SnapshotEstimator,
    compute_mc_num_samples,
    present_to_mask_int,
)

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class PermutationOptions(PluginOptions):
    """Permutation sampling options."""

    antithetic: bool = True
    """Pair every ordering with its reverse."""


class PermutationEstimator(_SnapshotEstimator):
    """Sample orderings, evaluate their prefixes and delegate to the estimand."""

    options_model: ClassVar[type[PluginOptions]] = PermutationOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read ``antithetic`` from the plugin options in ``context``."""
        super().__init__(context)
        options = PermutationOptions.model_validate(context.options if context else {})
        self.antithetic = options.antithetic

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "permutation"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield the grand and empty coalitions, then the prefixes of each ordering.

        Prefixes already yielded are skipped, so the count never exceeds the budget.
        """
        num_players = player_set.num_players
        budget = compute_mc_num_samples(
            num_players, budget_fraction, include_minimal_masks
        )
        grand = (True,) * num_players
        empty = (False,) * num_players
        seen = {present_to_mask_int(grand), present_to_mask_int(empty)}
        yield CoalitionMask.from_sequence(grand)
        yield CoalitionMask.from_sequence(empty)
        rng = np.random.default_rng(seed)
        yielded = 2
        while yielded < budget:
            order = rng.permutation(num_players)
            orders = [order, order[::-1]] if self.antithetic else [order]
            for ordering in orders:
                present = [False] * num_players
                for player in ordering[:-1]:
                    present[int(player)] = True
                    key = present_to_mask_int(present)
                    if key in seen or yielded >= budget:
                        continue
                    seen.add(key)
                    yielded += 1
                    yield CoalitionMask.from_sequence(tuple(present))
