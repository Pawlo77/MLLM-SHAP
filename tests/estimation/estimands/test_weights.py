"""Tests for coalition weight formulas used by estimand aggregators.

@see [[src/mllm_shap/estimation/estimands/_weights.py]]
"""

import logging
import math
from collections.abc import Callable

import pytest

from mllm_shap.estimation.estimands._weights import (
    aggregate_from_marginals,
    banzhaf_weight,
    shapley_weight,
)
from mllm_shap.estimation.estimands.banzhaf import BanzhafAggregator
from mllm_shap.estimation.estimands.shapley import ShapleyAggregator

GAME: dict[tuple[bool, bool, bool], float] = {
    (False, False, False): 0.0,
    (True, False, False): 1.0,
    (False, True, False): 2.0,
    (False, False, True): 4.0,
    (True, True, False): 5.0,
    (True, False, True): 7.0,
    (False, True, True): 6.0,
    (True, True, True): 11.0,
}
"""Non-additive three-player game: players 0 and 2 gain 2 together, 0 and 1 gain 2."""


@pytest.mark.parametrize(
    ("coalition_size", "num_players", "expected"),
    [
        (0, 3, math.factorial(0) * math.factorial(2) / math.factorial(3)),
        (1, 3, math.factorial(1) * math.factorial(1) / math.factorial(3)),
        (2, 3, math.factorial(2) * math.factorial(0) / math.factorial(3)),
    ],
)
def test_shapley_coalition_weight(
    coalition_size: int,
    num_players: int,
    expected: float,
) -> None:
    """Shapley weight matches k!(n-k-1)!/n!."""
    aggregator = ShapleyAggregator()
    assert aggregator.coalition_weight(coalition_size, num_players) == pytest.approx(
        expected
    )


def test_banzhaf_coalition_weight_is_uniform() -> None:
    """Banzhaf assigns the same weight to every coalition size."""
    aggregator = BanzhafAggregator()
    num_players = 4
    expected = 1.0 / (2 ** (num_players - 1))
    for coalition_size in range(num_players):
        assert aggregator.coalition_weight(
            coalition_size, num_players
        ) == pytest.approx(expected)


def _aggregate(
    masks: list[tuple[bool, bool, bool]],
    weight_fn: Callable[[int, int], float],
) -> list[float]:
    return aggregate_from_marginals(masks, [GAME[mask] for mask in masks], weight_fn)


@pytest.mark.parametrize(
    ("weight_fn", "expected"),
    [
        # Player 0 marginals by |S|: 1; (5 - 2, 7 - 4) -> 3; 11 - 6 = 5.
        # Player 1: 2; (5 - 1, 6 - 4) -> 3; 11 - 7 = 4.
        # Player 2: 4; (7 - 1, 6 - 2) -> 5; 11 - 5 = 6.
        # Shapley: each stratum has probability 1/3: (1 + 3 + 5) / 3 = 3, ...
        (shapley_weight, [3.0, 3.0, 5.0]),
        # Banzhaf: strata 0, 1, 2 have probability 1/4, 1/2, 1/4:
        # 1/4 + 3/2 + 5/4 = 3, 2/4 + 3/2 + 4/4 = 3, 4/4 + 5/2 + 6/4 = 5.
        (banzhaf_weight, [3.0, 3.0, 5.0]),
    ],
)
def test_aggregate_from_marginals_on_full_table(
    weight_fn: Callable[[int, int], float], expected: list[float]
) -> None:
    assert _aggregate(list(GAME), weight_fn) == pytest.approx(expected, abs=1e-12)


# The partial table misses {2}: player 0 loses the pair ({2}, {0, 2}), player 1
# loses ({2}, {1, 2}) and player 2 loses its whole size-0 stratum (∅, {2}).
PARTIAL = [mask for mask in GAME if mask != (False, False, True)]


def test_aggregate_from_marginals_partial_table_shapley() -> None:
    # Stratum l: mean v of sampled size-(l + 1) coalitions with i minus mean v of
    # sampled size-l coalitions without i.
    # Player 0: 1 - 0; (5 + 7) / 2 - 2 = 4; 11 - 6 = 5: (1 + 4 + 5) / 3 = 10/3.
    # Player 1: 2; (5 + 6) / 2 - 1 = 4.5; 11 - 7 = 4: 10.5 / 3 = 3.5.
    # Player 2: stratum 0 has no size-1 coalition with it; (7 + 6) / 2 - (1 + 2) / 2
    #   = 5; 11 - 5 = 6: (5 + 6) / 2 = 5.5.
    assert _aggregate(PARTIAL, shapley_weight) == pytest.approx(
        [10 / 3, 3.5, 5.5], abs=1e-12
    )


def test_aggregate_from_marginals_partial_table_banzhaf() -> None:
    # Stratum probabilities C(2, k) / 4 = 1/4, 1/2, 1/4 on the differences above.
    # Player 0: 1/4 * 1 + 1/2 * 4 + 1/4 * 5 = 3.5.
    # Player 1: 1/4 * 2 + 1/2 * 4.5 + 1/4 * 4 = 3.75.
    # Player 2: (1/2 * 5 + 1/4 * 6) / (1/2 + 1/4) = 16/3.
    assert _aggregate(PARTIAL, banzhaf_weight) == pytest.approx(
        [3.5, 3.75, 16 / 3], abs=1e-12
    )


def test_aggregate_from_marginals_zero_fills_unobserved_players(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only ∅, {0}, {1}: player 2 is never present, so no stratum has a with-mean.
    masks = [(False, False, False), (True, False, False), (False, True, False)]
    with caplog.at_level(logging.WARNING):
        values = _aggregate(masks, banzhaf_weight)
    assert values == [1.0, 2.0, 0.0]
    assert caplog.messages == [
        (
            "1 of 3 players have no stratum sampled with and without them; "
            "their values are 0. Raise the budget. Players: [2]"
        )
    ]


@pytest.mark.parametrize(
    ("masks", "payoffs", "message"),
    [
        ([], [], "masks must not be empty"),
        ([(True,)], [1.0, 2.0], "masks and payoffs must have the same length"),
        ([()], [1.0], "masks must include at least one player"),
        (
            [(True, False), (True,)],
            [1.0, 2.0],
            "all masks must have the same number of players",
        ),
    ],
)
def test_aggregate_from_marginals_rejects_malformed_tables(
    masks: list[tuple[bool, ...]], payoffs: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        aggregate_from_marginals(masks, payoffs, shapley_weight)
