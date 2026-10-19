"""Hot LRU cache for coalition generation results.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_store.py]]
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..domain.generation import ForcedScore


@dataclass(frozen=True, slots=True)
class StoredGeneration:
    """One cached generation with the extras value functions score."""

    text: str
    logprobs: tuple[float, ...] | None = None
    forced: Mapping[str, ForcedScore] = field(default_factory=dict)
    audio_path: str | None = None


@dataclass
class HotResultStore:
    """In-memory LRU store for recently generated coalition outputs."""

    maxsize: int | None = 128
    """Maximum retained results; ``None`` disables eviction."""

    _values: OrderedDict[str, StoredGeneration] = field(default_factory=OrderedDict)
    """LRU mapping from coalition key to the generation."""

    def get(self, coalition_key: str) -> str | None:
        """Return a cached generation or ``None`` when absent."""
        if coalition_key not in self._values:
            return None
        self._values.move_to_end(coalition_key)
        return self._values[coalition_key].text

    def logprobs(self, coalition_key: str) -> tuple[float, ...] | None:
        """Return the cached generation's token logprobs, ``None`` when unknown."""
        entry = self._values.get(coalition_key)
        return None if entry is None else entry.logprobs

    def entry(self, coalition_key: str) -> StoredGeneration | None:
        """Return the full cached generation, ``None`` when absent."""
        return self._values.get(coalition_key)

    def put(
        self,
        coalition_key: str,
        generation_text: str,
        logprobs: tuple[float, ...] | None = None,
        forced: Mapping[str, ForcedScore] | None = None,
        audio_path: str | None = None,
    ) -> None:
        """Insert or refresh a cached generation result."""
        if coalition_key in self._values:
            self._values.move_to_end(coalition_key)
        self._values[coalition_key] = StoredGeneration(
            generation_text, logprobs, forced or {}, audio_path
        )
        if self.maxsize is None:
            return
        while len(self._values) > self.maxsize:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        """Return the number of cached coalition results."""
        return len(self._values)
