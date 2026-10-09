"""Tests for prefix-cache helpers.

@see [[src/nlp_shap/runtime/kv_cache.py]]
"""

from nlp_shap.domain.coalition import CoalitionMask
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.masking.partitions import TokenPartitioner
from nlp_shap.runtime.kv_cache import (
    CroppableCache,
    PrefixCacheManager,
    build_snapshot_prefix_hash,
    group_jobs_for_prefix_cache,
)
from nlp_shap.runtime.scheduler import CoalitionJob


def test_build_snapshot_prefix_hash_ignores_final_player() -> None:
    """Coalitions that differ only in the last player share a prefix hash."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="shared alpha beta"),)),
    ))
    players = TokenPartitioner().partition(snapshot)

    def prefix(*present: bool) -> str:
        return build_snapshot_prefix_hash(
            snapshot, players, CoalitionMask.from_sequence(present)
        )

    assert prefix(True, False, True) == prefix(True, False, False)
    assert prefix(True, False, True) != prefix(True, True, True)


def test_longest_prefix_returns_longest_match() -> None:
    """The longest stored proper prefix wins and counts as a hit."""
    cache = PrefixCacheManager(max_entries=4)
    cache.store((1, 2), "short")
    cache.store((1, 2, 3), "long")
    cache.store((9, 9, 9), "other")

    assert cache.longest_prefix((1, 2, 3, 4)) == ("long", 3)
    assert cache.longest_prefix((1, 2, 5)) == ("short", 2)
    assert cache.longest_prefix((7, 8)) == (None, 0)
    assert (cache.hits, cache.misses) == (2, 1)


def test_longest_prefix_ignores_full_match() -> None:
    """A stored key equal to the prompt leaves no token to generate from."""
    cache = PrefixCacheManager()
    cache.store((1, 2, 3), "state")
    assert cache.longest_prefix((1, 2, 3)) == (None, 0)


class _Croppable:
    def __init__(self) -> None:
        self.length: int | None = None

    def crop(self, max_length: int) -> None:
        self.length = max_length


def test_longest_prefix_matches_common_prefix_of_croppable_states() -> None:
    """Croppable states also serve prompts that diverge before the key ends."""
    cache = PrefixCacheManager()
    state = _Croppable()
    cache.store((1, 2, 3, 4), state)

    assert cache.longest_prefix((1, 2, 9, 9)) == (state, 2)
    assert cache.longest_prefix((1, 2, 3, 4)) == (state, 3)
    assert isinstance(state, CroppableCache)


def test_store_evicts_oldest_beyond_max_entries() -> None:
    cache = PrefixCacheManager(max_entries=2)
    cache.store((1,), "a")
    cache.store((2,), "b")
    assert cache.longest_prefix((1, 5)) == ("a", 1)  # refreshes (1,)
    cache.store((3,), "c")

    assert cache.longest_prefix((2, 5)) == (None, 0)
    assert cache.longest_prefix((1, 5)) == ("a", 1)
    assert cache.longest_prefix((3, 5)) == ("c", 1)


def test_group_jobs_for_prefix_cache_sorts_by_prefix_hash(
    sample_snapshot,
) -> None:
    """Prefix grouping keeps coalitions with the same hash adjacent."""
    jobs = [
        CoalitionJob(
            coalition_key="b",
            snapshot_id=sample_snapshot.snapshot_id,
            snapshot=sample_snapshot,
            absence_policy="pad",
            mask_words=b"\x01",
            mask_n_bits=1,
            prefix_hash="beta",
            model_id="mock",
        ),
        CoalitionJob(
            coalition_key="a",
            snapshot_id=sample_snapshot.snapshot_id,
            snapshot=sample_snapshot,
            absence_policy="pad",
            mask_words=b"\x02",
            mask_n_bits=1,
            prefix_hash="alpha",
            model_id="mock",
        ),
    ]

    grouped = group_jobs_for_prefix_cache(jobs, key=lambda job: job.prefix_hash)

    assert [job.prefix_hash for job in grouped] == ["alpha", "beta"]
