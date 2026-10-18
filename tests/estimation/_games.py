"""Cooperative-game fixtures shared by estimator and axiom tests."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Game:
    """A cooperative game with its exact Shapley and Banzhaf values."""

    name: str
    num_players: int
    payoff: Callable[[tuple[bool, ...]], float]
    shapley: np.ndarray
    banzhaf: np.ndarray

    def table(self) -> tuple[list[tuple[bool, ...]], list[float]]:
        """Return the full coalition table and its payoffs."""
        masks = full_table(self.num_players)
        return masks, [self.payoff(mask) for mask in masks]


def _brute_force(num_players: int, payoff: Callable[[tuple[bool, ...]], float]) -> Game:
    """Return the game with Shapley and Banzhaf values from the full table."""
    values = {mask: payoff(mask) for mask in full_table(num_players)}
    shapley = np.zeros(num_players)
    banzhaf = np.zeros(num_players)
    for mask, value in values.items():
        size = sum(mask)
        for i, present in enumerate(mask):
            if present:
                marginal = value - values[(*mask[:i], False, *mask[i + 1 :])]
                shapley[i] += (
                    marginal
                    * math.factorial(size - 1)
                    * math.factorial(num_players - size)
                    / math.factorial(num_players)
                )
                banzhaf[i] += marginal / 2 ** (num_players - 1)
    return Game("", num_players, payoff, shapley, banzhaf)


def additive_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return ``v(S) = sum of weights``; both indices equal the weights."""
    weights = np.random.default_rng(seed).uniform(-1.0, 2.0, num_players)
    return Game(
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
    return Game("pairwise", num_players, payoff, exact, exact)


def airport_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return the airport game ``v(S) = max cost``; Shapley by the closed form."""
    costs = np.sort(np.random.default_rng(seed).uniform(1.0, 10.0, num_players))
    steps = np.diff(costs, prepend=0.0)
    shapley = np.array([
        sum(steps[j] / (num_players - j) for j in range(k + 1))
        for k in range(num_players)
    ])
    banzhaf = _brute_force(
        num_players,
        lambda mask: max(
            (c for c, m in zip(costs, mask, strict=True) if m), default=0.0
        ),
    ).banzhaf
    return Game(
        "airport",
        num_players,
        lambda mask: max(
            (c for c, m in zip(costs, mask, strict=True) if m), default=0.0
        ),
        shapley,
        banzhaf,
    )


def voting_game(num_players: int = 10, seed: int = 0) -> Game:
    """Return a weighted majority game; exact values by enumeration."""
    weights = np.random.default_rng(seed).integers(1, 6, num_players)
    quota = weights.sum() / 2

    def payoff(mask: tuple[bool, ...]) -> float:
        return float(weights[list(mask)].sum() > quota)

    game = _brute_force(num_players, payoff)
    return Game("voting", num_players, payoff, game.shapley, game.banzhaf)


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

    return Game("unanimity", num_players, payoff, shapley, banzhaf)


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

    game = _brute_force(num_players, payoff)
    return Game("mock_prompt", num_players, payoff, game.shapley, game.banzhaf)


def reference_games(num_players: int = 10) -> list[Game]:
    """Return every reference game with ``num_players`` players."""
    return [
        additive_game(num_players),
        pairwise_game(num_players),
        airport_game(num_players),
        voting_game(num_players),
        unanimity_game(num_players),
        mock_prompt_game(num_players),
    ]
