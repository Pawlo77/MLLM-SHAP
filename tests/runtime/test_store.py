"""Tests for hot LRU result store.

@see [[src/mllm_shap/runtime/store.py]]
"""

from mllm_shap.runtime.store import HotResultStore


def test_hot_store_returns_cached_generation() -> None:
    """Stored coalition results are returned on subsequent lookups."""
    store = HotResultStore(maxsize=2)
    store.put("k1", "first")
    assert store.get("k1") == "first"
    assert store.get("missing") is None


def test_hot_store_evicts_least_recently_used_entry() -> None:
    """Entries beyond maxsize evict the oldest coalition result."""
    store = HotResultStore(maxsize=2)
    store.put("k1", "one")
    store.put("k2", "two")
    store.get("k1")
    store.put("k3", "three")
    assert store.get("k2") is None
    assert store.get("k1") == "one"
    assert store.get("k3") == "three"


def test_hot_store_unbounded_when_maxsize_none() -> None:
    """``maxsize=None`` retains every inserted coalition result."""
    store = HotResultStore(maxsize=None)
    for index in range(200):
        store.put(f"k{index}", f"v{index}")
    assert len(store) == 200
    assert store.get("k0") == "v0"
    assert store.get("k199") == "v199"
