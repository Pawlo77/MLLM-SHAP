"""Attribution normalizer protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Sequence
from typing import Protocol


class Normalizer(Protocol):
    """Scale aggregated attributions for presentation without mutating archives."""

    @property
    def name(self) -> str:
        """Return the registered normalizer identifier."""

    def normalize(self, values: Sequence[float]) -> list[float]:
        """Return presentation-scaled copies of ``values``."""
