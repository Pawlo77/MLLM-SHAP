"""Attribution renderer protocol.

@domain protocols
@see [[knowledge/architecture/protocols]]
"""

from collections.abc import Sequence
from typing import Any, Protocol

from ..domain.estimands import Estimand


class AttributionRenderer(Protocol):
    """Render per-player attribution values for human review."""

    @property
    def name(self) -> str:
        """Return the registered renderer identifier."""

    def render(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        estimand: Estimand,
        title: str | None = None,
    ) -> Any:
        """Return a visualization artifact for the given labels and values."""
