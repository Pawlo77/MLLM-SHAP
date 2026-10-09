"""Prefix-cache storage for shared prompt token prefixes.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_kv_cache.py]]
"""

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet


def build_snapshot_prefix_hash(
    snapshot: ConversationSnapshot,
    player_set: PlayerSet,
    mask: CoalitionMask,
) -> str:
    """Hash the coalition prefix: the presence of every player except the last.

    Coalitions of ``snapshot`` that differ only in the last player render the
    same text up to that player's span, so they share a prompt prefix and run
    next to each other for KV-cache reuse. Nothing is re-tokenised.
    """
    hasher = hashlib.sha256(snapshot.snapshot_id.encode("utf-8"))
    for player_id, present in zip(
        player_set.player_ids[:-1], mask.present[:-1], strict=True
    ):
        hasher.update(player_id.encode("utf-8"))
        hasher.update(b"\x01" if present else b"\x00")
    return hasher.hexdigest()[:16]


@runtime_checkable
class CroppableCache(Protocol):
    """KV state that can be cut back to its first tokens (HF ``DynamicCache``)."""

    def crop(self, max_length: int) -> None:
        """Keep the state of the first ``max_length`` tokens only."""


class PrefixCacheManager:
    """Keep at most ``max_entries`` prefix KV states keyed by token-id tuple."""

    def __init__(self, max_entries: int = 8) -> None:
        # key -> (state, whether the state is a CroppableCache)
        """Create an empty cache holding at most ``max_entries`` states."""
        self._entries: OrderedDict[tuple[int, ...], tuple[Any, bool]] = OrderedDict()
        self._max_entries = max_entries
        self.hits = 0
        """Number of lookups that found a cached prefix."""
        self.misses = 0
        """Number of lookups that found nothing."""

    def longest_prefix(self, token_ids: Sequence[int]) -> tuple[Any | None, int]:
        """Return the cached state sharing the longest token prefix with ``token_ids``.

        A stored key matches when it is a proper prefix of ``token_ids``, or, for
        :class:`CroppableCache` states, when it shares at least one leading token
        (the caller crops the state to the returned length). At least one token
        is always left to process. Walks every stored entry once.
        """
        ids = tuple(int(t) for t in token_ids)
        best_len, best = 0, None
        best_key: tuple[int, ...] = ()
        for key, (value, croppable) in self._entries.items():
            if croppable:
                shared = min(_common_prefix_length(key, ids), len(ids) - 1)
            elif len(key) < len(ids) and ids[: len(key)] == key:
                shared = len(key)
            else:
                continue
            if shared > best_len:
                best_len, best, best_key = shared, value, key
        if best is None:
            self.misses += 1
            return None, 0
        self.hits += 1
        self._entries.move_to_end(best_key)
        return best, best_len

    def store(self, token_ids: Sequence[int], past_key_values: Any) -> None:
        """Store the KV state of ``token_ids``, evicting the oldest beyond the cap."""
        key = tuple(int(t) for t in token_ids)
        self._entries[key] = (
            past_key_values,
            isinstance(past_key_values, CroppableCache),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def reset(self) -> None:
        """Clear cached prefixes and counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0


def _common_prefix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the length of the shared leading run of two token tuples.

    Uses tuple slice equality (C speed) with a binary search on the length.
    """
    limit = min(len(left), len(right))
    if left[:limit] == right[:limit]:
        return limit
    low, high = 0, limit  # left[:low] matches; left[:high] does not
    while high - low > 1:
        middle = (low + high) // 2
        if left[:middle] == right[:middle]:
            low = middle
        else:
            high = middle
    return low


def group_jobs_for_prefix_cache[T](
    jobs: list[T],
    *,
    key: Callable[[T], str],
) -> list[T]:
    """Sort jobs so coalitions with the same prefix hash run adjacently."""
    return sorted(jobs, key=key)
