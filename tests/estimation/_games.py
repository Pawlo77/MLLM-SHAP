"""Cooperative-game fixtures shared by estimator and axiom tests."""

from collections.abc import Sequence

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
