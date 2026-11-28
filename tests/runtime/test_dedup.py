"""Tests for coalition deduplication keys and registry.

@see [[src/mllm_shap/runtime/dedup.py]]
"""

import pytest

from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role
from mllm_shap.masking.partitions import TokenPartitioner
from mllm_shap.pipeline.config import DedupConfig, GenerationConfig
from mllm_shap.runtime.dedup import (
    CoalitionDedupRegistry,
    build_coalition_key,
    dedup_enabled,
)

GOLDEN_SNAPSHOTS = {
    "text": ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="Be brief."),
                Message(role=Role.USER, text="Is the sky blue today?"),
            )
        ),
    )),
    "audio": ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(
                    role=Role.USER,
                    content=(
                        TextPart("What is said?"),
                        AudioPart(bytes(i % 251 for i in range(4000)), 16_000),
                    ),
                    speaker="ann",
                ),
            )
        ),
    )),
}
"""Snapshots whose ids, player ids and coalition keys are pinned at 0.10.0."""

GOLDEN_IDS = {"text": ("f8f38bd1093ef0ff", 7), "audio": ("8ea61b6b1279ecd5", 3)}
"""Snapshot id and token-player count of each golden snapshot at 0.10.0."""

SAMPLED = GenerationConfig(temperature=0.7, top_k=40, max_new_tokens=48)
"""A sampled generation setting, so every hashed generation field is non-default."""


def test_golden_snapshot_and_player_ids() -> None:
    """Snapshot ids and token player ids keep their 0.10.0 values."""
    for name, (snapshot_id, num_players) in GOLDEN_IDS.items():
        snapshot = GOLDEN_SNAPSHOTS[name]
        assert snapshot.snapshot_id == snapshot_id
        players = TokenPartitioner().partition(snapshot)
        assert players.num_players == num_players
        assert players.player_ids[:2] == (
            f"{snapshot_id}:0:0:0",
            f"{snapshot_id}:0:0:1",
        )


@pytest.mark.parametrize(
    ("name", "policy", "generation", "draw", "key"),
    [
        (
            "text",
            "delete",
            GenerationConfig(),
            0,
            "9ff88b02c27950013e194873644729b95312c0b66490efa93b26b7b24b98a335",
        ),
        (
            "text",
            "pad",
            GenerationConfig(),
            0,
            "3eb3d72df81922c197e787c7f006411f51f68b7b9dd73ba1c0f0a4ff28227f16",
        ),
        (
            "text",
            "delete",
            GenerationConfig(),
            2,
            "09e27bfd53b56072f5460316ff135f351575b966be7254e119d4e04b84f356a2",
        ),
        (
            "text",
            "delete",
            SAMPLED,
            0,
            "fd870f3d4b1eb648f369487bc0691436a60aebe4ab79916c071bdca5d9416d2a",
        ),
        (
            "audio",
            "delete",
            GenerationConfig(),
            0,
            "eb84ee09d0f0dbe46d55fef501a2b7f71009258cf7de91be311bfbf83506d62d",
        ),
        (
            "audio",
            "pad",
            GenerationConfig(),
            0,
            "e650681e328edd37ecbb956aec9cdc908f32305457df916ccd392c20215f54a8",
        ),
        (
            "audio",
            "delete",
            GenerationConfig(),
            2,
            "fcafa9f9394fad1e666b8e93057318495b8785f0c5d54cd14fdc818ff536ad52",
        ),
        (
            "audio",
            "delete",
            SAMPLED,
            0,
            "f0f3b493f7be314812b371c49905d4d1ad17917acbd35f043d439bf83ad1e6f6",
        ),
    ],
)
def test_golden_coalition_keys(
    name: str, policy: str, generation: GenerationConfig, draw: int, key: str
) -> None:
    """Coalition keys keep their 0.10.0 values, so earlier archives still replay."""
    snapshot = GOLDEN_SNAPSHOTS[name]
    players = TokenPartitioner().partition(snapshot)
    mask = tuple(index % 2 == 0 for index in range(players.num_players))
    assert (
        build_coalition_key(
            snapshot_id=snapshot.snapshot_id,
            player_ids=players.player_ids,
            mask_present=mask,
            absence_policy=policy,
            model_id="org/model",
            generation=generation,
            draw=draw,
        )
        == key
    )


def test_build_coalition_key_is_stable() -> None:
    """Identical coalition inputs produce the same SHA256 key."""
    generation = GenerationConfig(temperature=0.0)
    first = build_coalition_key(
        snapshot_id="snap",
        player_ids=("a", "b"),
        mask_present=(True, False),
        absence_policy="delete",
        model_id="mock",
        generation=generation,
    )
    second = build_coalition_key(
        snapshot_id="snap",
        player_ids=("a", "b"),
        mask_present=(True, False),
        absence_policy="delete",
        model_id="mock",
        generation=generation,
    )
    assert first == second
    assert len(first) == 64


def test_build_coalition_key_changes_with_generation() -> None:
    """Different generation settings produce different coalition keys."""
    base_kwargs = {
        "snapshot_id": "snap",
        "player_ids": ("a",),
        "mask_present": (True,),
        "absence_policy": "delete",
        "model_id": "mock",
    }
    cold = build_coalition_key(
        generation=GenerationConfig(temperature=0.0),
        **base_kwargs,
    )
    warm = build_coalition_key(
        generation=GenerationConfig(temperature=0.7),
        **base_kwargs,
    )
    assert cold != warm


def test_dedup_enabled_modes() -> None:
    """Dedup auto mode is on at any temperature, so repeated masks share a payoff."""
    generation = GenerationConfig(temperature=0.0)
    assert dedup_enabled(DedupConfig(enabled="on"), generation) is True
    assert dedup_enabled(DedupConfig(enabled="off"), generation) is False
    assert dedup_enabled(DedupConfig(enabled="auto"), generation) is True
    assert (
        dedup_enabled(
            DedupConfig(enabled="auto"),
            GenerationConfig(temperature=0.5),
        )
        is True
    )


def test_registry_tracks_unique_keys() -> None:
    """Repeated coalition keys are observed only once."""
    registry = CoalitionDedupRegistry()
    assert registry.observe("alpha") is True
    assert registry.observe("alpha") is False
    assert registry.observe("beta") is True
    assert registry.contains("alpha") is True
    assert len(registry) == 2


def test_policy_options_extend_the_key_only_when_set() -> None:
    """Empty options keep the old key; each option value gives its own key."""
    args = {
        "snapshot_id": "snap",
        "player_ids": ("a", "b"),
        "mask_present": (True, False),
        "absence_policy": "pad",
        "model_id": "mock",
        "generation": GenerationConfig(),
    }
    default = build_coalition_key(**args)
    assert build_coalition_key(**args, policy_options={}) == default
    first = build_coalition_key(**args, policy_options={"placeholder": "<x>"})
    second = build_coalition_key(**args, policy_options={"placeholder": "<y>"})
    assert len({default, first, second}) == 3
