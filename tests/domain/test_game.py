"""Tests for cooperative-game domain types.

@see [[src/mllm_shap/domain/game.py]]
"""

from mllm_shap.domain.game import CooperativeGame
from mllm_shap.domain.players import PlayerSet


def test_cooperative_game_stores_player_set() -> None:
    """Cooperative games retain the player set."""
    players = PlayerSet(player_ids=("t0", "t1"))
    game = CooperativeGame(player_set=players)
    assert game.player_set is players
