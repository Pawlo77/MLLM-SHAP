"""Optional observability hooks for explain pipeline stages.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_telemetry.py]]
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SpanRecord:
    """One completed pipeline stage span."""

    name: str
    """Stage identifier emitted by the orchestrator or estimator."""

    duration_ms: float
    """Wall-clock duration of the stage in milliseconds."""


class ObservabilitySink(Protocol):
    """Protocol for recording explain pipeline stage spans."""

    @contextmanager
    def span(self, name: str) -> Generator[None]:
        """Record wall-clock duration for one named pipeline stage."""


class NullObservabilitySink:
    """No-op observability sink used when telemetry is disabled."""

    @contextmanager
    def span(self, name: str) -> Generator[None]:
        """Yield without recording span metadata."""
        _ = name
        yield


@dataclass
class InMemoryObservabilitySink:
    """Collect span records in memory for tests and diagnostics."""

    spans: list[SpanRecord] = field(default_factory=list)

    @contextmanager
    def span(self, name: str) -> Generator[None]:
        """Append one span record when the context exits."""
        started = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            self.spans.append(SpanRecord(name=name, duration_ms=duration_ms))
