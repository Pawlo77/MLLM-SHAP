"""Tests for explain pipeline telemetry.

@see [[src/nlp_shap/runtime/telemetry.py]]
"""

from nlp_shap.runtime.telemetry import InMemoryObservabilitySink


def test_in_memory_observability_sink_records_span_durations() -> None:
    """Span context manager appends one record per completed stage."""
    sink = InMemoryObservabilitySink()
    with sink.span("stage.one"):
        pass
    with sink.span("stage.two"):
        pass

    assert [span.name for span in sink.spans] == ["stage.one", "stage.two"]
    assert all(span.duration_ms >= 0.0 for span in sink.spans)
