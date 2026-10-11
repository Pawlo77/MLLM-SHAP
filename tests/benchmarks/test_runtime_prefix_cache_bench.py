"""Micro-benchmarks for prefix cache lookups.

@see [[src/mllm_shap/runtime/kv_cache.py]]
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mllm_shap.runtime.kv_cache import PrefixCacheManager

_PREFIXES = [tuple(range(length)) for length in range(4, 20)]


def _lookup_10k() -> int:
    cache = PrefixCacheManager(max_entries=16)
    for prefix in _PREFIXES:
        cache.store(prefix, object())
    hits = 0
    for index in range(10_000):
        token_ids = _PREFIXES[index % len(_PREFIXES)] + (index % 7,)
        matched, length = cache.longest_prefix(token_ids)
        if length > 0 and matched is not None:
            hits += 1
    return hits


@pytest.mark.bench
def test_prefix_cache_hits_10k(benchmark: BenchmarkFixture) -> None:
    """10k longest-prefix lookups on shared prefixes all hit."""
    assert benchmark(_lookup_10k) == 10_000
