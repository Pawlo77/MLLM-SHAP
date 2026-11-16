"""LIME baseline: a locally weighted ridge regression on coalition masks.

The surrogate of Ribeiro et al. (2016) for text: coalitions are random subsets, each
weighted by an exponential kernel of its cosine distance from the full input, and the
payoffs are regressed on presence with an unpenalised intercept. Values are
regression coefficients, not Shapley or Banzhaf values, so the estimand is ignored
and they need not add up to ``v(N) - v(empty)``.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_lime.py]]
"""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from ..protocols.plugin import PluginOptions
from ._shared import (
    _SnapshotEstimator,
    compute_mc_num_samples,
    present_to_mask_int,
)

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class LimeOptions(PluginOptions):
    """Surrogate settings, with the defaults of the reference ``lime`` package."""

    kernel_width: float = 25.0
    """Width of the exponential kernel over cosine distance times 100."""

    alpha: float = 1.0
    """Ridge penalty on the coefficients."""


def lime_values(
    present: np.ndarray,
    payoffs: Sequence[float],
    kernel_width: float = 25.0,
    alpha: float = 1.0,
) -> list[float]:
    """Return the ridge coefficients of payoffs on presence rows.

    Row weights are ``sqrt(exp(-d^2 / width^2))`` with ``d`` the cosine distance to the
    all-present row times 100; an all-absent row has distance 1.
    """
    rows = np.asarray(present, dtype=np.float64)
    pay = np.asarray(payoffs, dtype=np.float64)
    count = rows.sum(axis=1)
    distance = 100.0 * (1.0 - np.sqrt(count / rows.shape[1]))
    weights = np.sqrt(np.exp(-(distance**2) / kernel_width**2))
    total = weights.sum()
    mean_x = weights @ rows / total
    centred = rows - mean_x
    target = pay - weights @ pay / total
    gram = centred.T @ (centred * weights[:, None]) + alpha * np.eye(rows.shape[1])
    solution = np.linalg.solve(gram, centred.T @ (weights * target))
    return [float(value) for value in solution]


class LimeEstimator(_SnapshotEstimator):
    """Sample random subsets and fit the LIME surrogate on their payoffs."""

    options_model: ClassVar[type[PluginOptions]] = LimeOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read the kernel width and penalty from the plugin options in ``context``."""
        super().__init__(context)
        options = LimeOptions.model_validate(context.options if context else {})
        self.kernel_width = options.kernel_width
        self.alpha = options.alpha

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "lime"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield the grand and empty coalitions, then distinct random subsets.

        The number of removed players is uniform over ``1..n-1``. Subsets are distinct,
        so ``reanalyze`` of the archive sees the same rows as the run.
        """
        n = player_set.num_players
        budget = compute_mc_num_samples(n, budget_fraction, include_minimal_masks)
        grand, empty = (True,) * n, (False,) * n
        seen = {present_to_mask_int(grand), present_to_mask_int(empty)}
        yield CoalitionMask.from_sequence(grand)
        yield CoalitionMask.from_sequence(empty)
        rng = np.random.default_rng(seed)
        attempts = 0
        while len(seen) < budget and n > 1 and attempts < 50 * budget:
            attempts += 1
            removed = rng.choice(n, size=int(rng.integers(1, n)), replace=False)
            present = [True] * n
            for index in removed:
                present[int(index)] = False
            key = present_to_mask_int(present)
            if key not in seen:
                seen.add(key)
                yield CoalitionMask.from_sequence(tuple(present))

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator | None,
    ) -> list[float]:
        """Return the surrogate coefficients; the estimand is ignored."""
        _ = aggregator
        rows = np.array([mask.present for mask in masks], dtype=np.float64)
        return lime_values(rows, payoffs, self.kernel_width, self.alpha)
