"""Hot LRU cache for coalition generation results.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_store.py]]
"""

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class HotResultStore:
    """In-memory LRU store for recently generated coalition outputs."""

    maxsize: int | None = 128
    """Maximum retained results; ``None`` disables eviction."""

    _values: OrderedDict[str, tuple[str, tuple[float, ...] | None]] = field(
        default_factory=OrderedDict
    )
    """LRU mapping from coalition key to generated text and token logprobs."""

    def get(self, coalition_key: str) -> str | None:
        """Return a cached generation or ``None`` when absent."""
        if coalition_key not in self._values:
            return None
        self._values.move_to_end(coalition_key)
        return self._values[coalition_key][0]

    def logprobs(self, coalition_key: str) -> tuple[float, ...] | None:
        """Return the cached generation's token logprobs, ``None`` when unknown."""
        entry = self._values.get(coalition_key)
        return None if entry is None else entry[1]

    def put(
        self,
        coalition_key: str,
        generation_text: str,
        logprobs: tuple[float, ...] | None = None,
    ) -> None:
        """Insert or refresh a cached generation result."""
        if coalition_key in self._values:
            self._values.move_to_end(coalition_key)
        self._values[coalition_key] = (generation_text, logprobs)
        if self.maxsize is None:
            return
        while len(self._values) > self.maxsize:
            self._values.popitem(last=False)

    def __len__(self) -> int:
        """Return the number of cached coalition results."""
        return len(self._values)
