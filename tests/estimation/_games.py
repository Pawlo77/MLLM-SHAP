"""Cooperative-game fixtures shared by estimator and axiom tests."""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hypothesis import strategies as st


def full_table(num_players: int) -> list[tuple[bool, ...]]:
    """Return every coalition mask for ``num_players`` ordered by bitmask value."""
    return [
        tuple(bool((value >> index) & 1) for index in range(num_players))
        for value in range(1 << num_players)
    ]


def additive_payoffs(
    masks: Sequence[Sequence[bool]],
    coefficients: Sequence[float],
) -> list[float]:
    """Return v(S) = sum of coefficients of present players."""
    return [
        sum(coef for coef, present in zip(coefficients, mask, strict=True) if present)
        for mask in masks
    ]


@st.composite
def random_game(
    draw: st.DrawFn,
    min_players: int = 1,
    max_players: int = 6,
) -> tuple[int, list[tuple[bool, ...]], list[float]]:
    """Draw a random characteristic function over a full coalition table."""
    num_players = draw(st.integers(min_value=min_players, max_value=max_players))
    masks = full_table(num_players)
    payoffs = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
            min_size=len(masks),
            max_size=len(masks),
        )
    )
    return num_players, masks, payoffs


BETA_SHAPLEY = (4.0, 1.0)
"""``(alpha, beta)`` of the Beta Shapley reference values."""

WEIGHTED_BANZHAF_P = 0.3
"""Inclusion probability of the weighted Banzhaf reference values."""


@dataclass(frozen=True)
class Game:
    """A cooperative game with its exact semivalues and pairwise interactions."""

    name: str
    num_players: int
    payoff: Callable[[tuple[bool, ...]], float]
    shapley: np.ndarray
    banzhaf: np.ndarray
    beta_shapley: np.ndarray
    weighted_banzhaf: np.ndarray
    interactions: np.ndarray
    """Symmetric matrix of Shapley interaction indices of order 2, zero diagonal."""

    def table(self) -> tuple[list[tuple[bool, ...]], list[float]]:
        """Return the full coalition table and its payoffs."""
        masks = full_table(self.num_players)
        return masks, [self.payoff(mask) for mask in masks]


def _semivalue(table: np.ndarray, num_players: int, weights: np.ndarray) -> np.ndarray:
    """Return ``sum_S w[|S|] (v(S + i) - v(S))`` per player over a bit-indexed table."""
    coalitions = np.arange(1 << num_players)
    sizes = np.array([int(c).bit_count() for c in coalitions])
    values = np.zeros(num_players)
    for i in range(num_players):
        without = coalitions[((coalitions >> i) & 1) == 0]
        marginal = table[without | (1 << i)] - table[without]
        values[i] = (weights[sizes[without]] * marginal).sum()
    return values


def _interactions(table: np.ndarray, num_players: int) -> np.ndarray:
    """Return the order-2 Shapley interaction index matrix by enumeration."""
    coalitions = np.arange(1 << num_players)
    sizes = np.array([int(c).bit_count() for c in coalitions])
    out = np.zeros((num_players, num_players))
    for i in range(num_players):
        for j in range(i + 1, num_players):
            both = (1 << i) | (1 << j)
            rest = coalitions[(coalitions & both) == 0]
            size = sizes[rest]
            weight = np.array([
                math.factorial(s) * math.factorial(num_players - s - 2) for s in size
            ]) / math.factorial(num_players - 1)
            delta = (
                table[rest | both]
                - table[rest | (1 << i)]
                - table[rest | (1 << j)]
                + table[rest]
            )
            out[i, j] = out[j, i] = (weight * delta).sum()
    return out


def _game(
    name: str,
    num_players: int,
    payoff: Callable[[tuple[bool, ...]], float],
    shapley: np.ndarray | None = None,
    banzhaf: np.ndarray | None = None,
) -> Game:
    """Return the game with every exact index enumerated; closed forms replace two.

    Shapley and Banzhaf may be given when a closed form exists, so the tests also
    check the enumeration against it.
    """
    table = np.array([payoff(mask) for mask in full_table(num_players)])
    sizes = np.arange(num_players)
    # w(s) is the weight of a coalition of size s without the player; each
    # semivalue's weights sum to one over all subsets of the other n - 1 players.
    log_beta = lambda a, b: math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)  # noqa: E731
    alpha, beta = BETA_SHAPLEY
    weights = {
        "shapley": np.array([
            math.factorial(s) * math.factorial(num_players - s - 1) for s in sizes
        ])
        / math.factorial(num_players),
        "banzhaf": np.full(num_players, 0.5 ** (num_players - 1)),
        "beta": np.exp([
            log_beta(s + beta, num_players - 1 - s + alpha) - log_beta(alpha, beta)
            for s in sizes
        ]),
        "weighted": WEIGHTED_BANZHAF_P**sizes
        * (1.0 - WEIGHTED_BANZHAF_P) ** (num_players - 1 - sizes),
    }
    exact = {k: _semivalue(table, num_players, w) for k, w in weights.items()}
    for given, key in ((shapley, "shapley"), (banzhaf, "banzhaf")):
        if given is not None:
            assert np.allclose(given, exact[key]), f"{name}: closed form differs"
    return Game(
        name,
        num_players,
        payoff,
        exact["shapley"],
        exact["banzhaf"],
        exact["beta"],
        exact["weighted"],
        _interactions(table, num_players),
    )


def additive_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return ``v(S) = sum of weights``; both indices equal the weights."""
    weights = np.random.default_rng(seed).uniform(-1.0, 2.0, num_players)
    return _game(
        "additive",
        num_players,
        lambda mask: float(weights[list(mask)].sum()),
        weights,
        weights,
    )


def pairwise_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return a pairwise-interaction game; both indices are ``a_i + sum_j w_ij / 2``."""
    rng = np.random.default_rng(seed)
    singles = rng.uniform(-1.0, 1.0, num_players)
    pairs = np.triu(rng.uniform(-1.0, 1.0, (num_players, num_players)), 1)
    sym = pairs + pairs.T

    def payoff(mask: tuple[bool, ...]) -> float:
        x = np.asarray(mask, dtype=float)
        return float(singles @ x + x @ pairs @ x)

    exact = singles + sym.sum(axis=1) / 2
    return _game("pairwise", num_players, payoff, exact, exact)


def airport_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return the airport game ``v(S) = max cost``; Shapley by the closed form."""
    costs = np.sort(np.random.default_rng(seed).uniform(1.0, 10.0, num_players))
    steps = np.diff(costs, prepend=0.0)
    shapley = np.array([
        sum(steps[j] / (num_players - j) for j in range(k + 1))
        for k in range(num_players)
    ])

    def payoff(mask: tuple[bool, ...]) -> float:
        return max((c for c, m in zip(costs, mask, strict=True) if m), default=0.0)

    return _game("airport", num_players, payoff, shapley)


def voting_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return a weighted majority game; exact values by enumeration."""
    weights = np.random.default_rng(seed).integers(1, 6, num_players)
    quota = weights.sum() / 2

    def payoff(mask: tuple[bool, ...]) -> float:
        return float(weights[list(mask)].sum() > quota)

    return _game("voting", num_players, payoff)


def unanimity_game(num_players: int = 10, seed: int = 0, terms: int = 12) -> Game:
    """Return a random sum of unanimity games with closed-form values.

    ``v(S) = sum_T c_T [T in S]`` gives ``phi_i = sum_{T has i} c_T / |T|`` and
    ``beta_i = sum_{T has i} c_T / 2^(|T| - 1)``.
    """
    rng = np.random.default_rng(seed)
    coalitions = [
        rng.choice(num_players, size=int(rng.integers(1, 5)), replace=False)
        for _ in range(terms)
    ]
    coefficients = rng.uniform(-1.0, 2.0, terms)
    shapley = np.zeros(num_players)
    banzhaf = np.zeros(num_players)
    for members, coefficient in zip(coalitions, coefficients, strict=True):
        shapley[members] += coefficient / len(members)
        banzhaf[members] += coefficient / 2 ** (len(members) - 1)

    def payoff(mask: tuple[bool, ...]) -> float:
        present = np.asarray(mask)
        return float(
            sum(
                c
                for m, c in zip(coalitions, coefficients, strict=True)
                if present[m].all()
            )
        )

    return _game("unanimity", num_players, payoff, shapley, banzhaf)


def mock_prompt_game(num_players: int = 12) -> Game:
    """Return a seeded 12-player table standing in for a recorded mock-backend prompt.

    Smooth sub-additive payoffs with a few interactions, the shape of
    similarity-based value functions; exact values by enumeration.
    """
    rng = np.random.default_rng(12)
    weights = rng.uniform(0.0, 1.0, num_players)
    pairs = np.triu(rng.uniform(-0.3, 0.3, (num_players, num_players)), 1)

    def payoff(mask: tuple[bool, ...]) -> float:
        x = np.asarray(mask, dtype=float)
        return float(np.tanh(weights @ x + x @ pairs @ x))

    return _game("mock_prompt", num_players, payoff)


RECORDED_TABLE = Path(__file__).parent / "fixtures" / "teacher_forced_10.json"
"""Ten-word teacher-forced table: the ``tf_logprob`` of every coalition of a prompt."""


def recorded_teacher_forced_game() -> Game:
    """Return the recorded 10-word teacher-forced game; exact values by enumeration.

    Recorded with ``hf-internal-testing/tiny-random-gpt2`` in float32, absence
    ``delete``: a real forward pass of a transformer with random weights, so its
    values are real model output but almost flat.
    """
    table = json.loads(RECORDED_TABLE.read_text(encoding="utf-8"))["payoffs"]

    def payoff(mask: tuple[bool, ...]) -> float:
        return float(table[sum(1 << i for i, present in enumerate(mask) if present)])

    return _game("teacher_forced", 10, payoff)


def reference_games(num_players: int = 10) -> list[Game]:
    """Return every reference game with ``num_players`` players."""
    games = [
        additive_game(num_players),
        pairwise_game(num_players),
        airport_game(num_players),
        voting_game(num_players),
        unanimity_game(num_players),
        mock_prompt_game(num_players),
    ]
    if num_players == 10:
        games.append(recorded_teacher_forced_game())
    return games
