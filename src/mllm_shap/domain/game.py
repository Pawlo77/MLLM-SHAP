"""Cooperative-game domain types.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_game.py]]
"""

from dataclasses import dataclass

from .players import PlayerSet


@dataclass(frozen=True, slots=True)
class CooperativeGame:
    """Player set and reference values for a characteristic function."""

    player_set: PlayerSet
    """Ordered explainability players that define the game."""
