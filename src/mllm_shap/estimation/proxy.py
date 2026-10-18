"""ProxySHAP (order 1): a boosted-tree proxy with exact values and a residual fix.

A leaf of the fitted trees is the game ``c [P in S, A disjoint from S]`` for the
players present (``P``) and absent (``A``) on its path, with closed-form Shapley
and Banzhaf values, so the proxy's values are exact sums over leaves. The values
of the residual ``v - proxy`` on the sampled coalitions are added back when that
is cheap, which makes the estimate exact on the full table.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_proxy.py]]
"""

import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from ..domain.coalition import CoalitionMask
from ..domain.estimands import Estimand
from ..domain.players import PlayerSet
from ..protocols.estimand import EstimandAggregator
from .leverage import LeverageEstimator

if TYPE_CHECKING:
    from sklearn.ensemble import GradientBoostingRegressor

    from ..plugins.context import PluginContext

CORRECTION_MAX_PLAYERS = 30
"""Fewer players than this always get the residual correction."""

CORRECTION_MIN_PER_PLAYER = 16
"""Larger budgets, in coalitions per player, get the correction at any size."""


def _boosting_regressor() -> "type[GradientBoostingRegressor]":
    """Return scikit-learn's regressor; imported here so the extra stays optional."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as error:
        msg = "estimator 'proxy' needs scikit-learn; install it with 'mllm-shap[proxy]'"
        raise ImportError(msg) from error
    return cast("type[GradientBoostingRegressor]", GradientBoostingRegressor)


def leaf_game_values(
    present: Sequence[int], absent: Sequence[int], estimand: Estimand
) -> tuple[float, float]:
    """Return the per-player value of a unit leaf for ``present`` and ``absent``.

    Returns the value of each present and of each absent player. Shapley gives
    ``(p-1)! a! / (p+a)!`` and minus ``p! (a-1)! / (p+a)!``; Banzhaf gives plus and
    minus ``1 / 2^(p+a-1)``.
    """
    p, a = len(present), len(absent)
    if estimand is Estimand.BANZHAF:
        value = 0.5 ** (p + a - 1) if p + a else 0.0
        return value, -value
    total = math.factorial(p + a) if p + a else 1
    gain = math.factorial(p - 1) * math.factorial(a) / total if p else 0.0
    loss = math.factorial(p) * math.factorial(a - 1) / total if a else 0.0
    return gain, -loss


def tree_values(
    model: "GradientBoostingRegressor", num_players: int, estimand: Estimand
) -> np.ndarray:
    """Return exact per-player values of the fitted ensemble's game."""
    values = np.zeros(num_players)
    for stage in model.estimators_[:, 0]:
        tree = stage.tree_
        stack: list[tuple[int, frozenset[int], frozenset[int]]] = [
            (0, frozenset(), frozenset())
        ]
        while stack:
            node, present, absent = stack.pop()
            left, right = tree.children_left[node], tree.children_right[node]
            if left == -1:
                gain, loss = leaf_game_values(sorted(present), sorted(absent), estimand)
                scale = model.learning_rate * float(tree.value[node, 0, 0])
                values[list(present)] += scale * gain
                values[list(absent)] += scale * loss
                continue
            feature = int(tree.feature[node])
            # A binary feature is constant below its own split, so a path never
            # tests it twice.
            stack.append((int(left), present, absent | {feature}))
            stack.append((int(right), present | {feature}, absent))
    return values


class ProxyEstimator(LeverageEstimator):
    """Leverage-sampled coalitions, a tree proxy, and a residual correction."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Fail early when scikit-learn is missing."""
        super().__init__(context)
        _boosting_regressor()

    @property
    def name(self) -> str:
        """Return the registered estimator identifier."""
        return "proxy"

    def sample_masks(
        self,
        player_set: PlayerSet,
        budget_fraction: float,
        include_minimal_masks: bool,
        seed: int,
    ) -> Iterator[CoalitionMask]:
        """Yield leverage-sampled complementary pairs."""
        return super().sample_masks(
            player_set, budget_fraction, include_minimal_masks, seed
        )

    def estimate_attributions(
        self,
        masks: Sequence[CoalitionMask],
        payoffs: Sequence[float],
        aggregator: EstimandAggregator,
    ) -> list[float]:
        """Fit the proxy on the archived coalitions and return its corrected values."""
        present = np.array([mask.present for mask in masks], dtype=np.float64)
        pay = np.asarray(payoffs, dtype=np.float64)
        num_players = present.shape[1]
        model = _boosting_regressor()(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0
        ).fit(present, pay)
        values = tree_values(model, num_players, aggregator.estimand)
        if (
            num_players < CORRECTION_MAX_PLAYERS
            or len(masks) >= CORRECTION_MIN_PER_PLAYER * num_players
        ):
            residual = pay - model.predict(present)
            values = values + np.array(
                aggregator.aggregate(
                    [mask.present for mask in masks], residual.tolist()
                )
            )
        return [float(value) for value in values]
