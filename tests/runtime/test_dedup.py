"""Tests for coalition deduplication keys and registry.

@see [[src/nlp_shap/runtime/dedup.py]]
"""

from nlp_shap.pipeline.config import DedupConfig, GenerationConfig
from nlp_shap.runtime.dedup import (
    CoalitionDedupRegistry,
    build_coalition_key,
    dedup_enabled,
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
    """Dedup auto mode follows deterministic generation temperature."""
    generation = GenerationConfig(temperature=0.0)
    assert dedup_enabled(DedupConfig(enabled="on"), generation) is True
    assert dedup_enabled(DedupConfig(enabled="off"), generation) is False
    assert dedup_enabled(DedupConfig(enabled="auto"), generation) is True
    assert (
        dedup_enabled(
            DedupConfig(enabled="auto"),
            GenerationConfig(temperature=0.5),
        )
        is False
    )


def test_registry_tracks_unique_keys() -> None:
    """Repeated coalition keys are observed only once."""
    registry = CoalitionDedupRegistry()
    assert registry.observe("alpha") is True
    assert registry.observe("alpha") is False
    assert registry.observe("beta") is True
    assert registry.contains("alpha") is True
    assert len(registry) == 2
