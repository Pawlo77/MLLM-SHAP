"""Micro-benchmarks for coalition deduplication.

@see [[src/mllm_shap/runtime/dedup.py]]
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mllm_shap.pipeline.config import GenerationConfig
from mllm_shap.runtime.dedup import CoalitionDedupRegistry, build_coalition_key

_GENERATION = GenerationConfig(max_new_tokens=32, temperature=0.0, top_k=1)
_PLAYER_IDS = tuple(f"p{index}" for index in range(16))


def _observe_10k() -> tuple[int, int]:
    registry = CoalitionDedupRegistry()
    duplicates = 0
    for index in range(10_000):
        mask = tuple(bool(((index % 100) >> bit) & 1) for bit in range(16))
        key = build_coalition_key(
            snapshot_id="bench-snapshot",
            player_ids=_PLAYER_IDS,
            mask_present=mask,
            absence_policy="pad",
            model_id="mock",
            generation=_GENERATION,
        )
        if not registry.observe(key):
            duplicates += 1
    return len(registry), duplicates


@pytest.mark.bench
def test_dedup_key_throughput_10k(benchmark: BenchmarkFixture) -> None:
    """Building and observing 10k coalition keys (100 unique)."""
    unique, duplicates = benchmark(_observe_10k)

    assert unique == 100
    assert duplicates == 10_000 - 100
