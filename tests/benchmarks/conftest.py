"""pytest-benchmark settings for the benchmark suite.

Benchmarks use the ``benchmark`` fixture of pytest-benchmark; ``make bench``
records a report and ``make bench-regression`` compares medians with the
committed ``baseline.json``.

@see [[tests/benchmarks/AGENTS.md]]
"""

import pytest


def pytest_benchmark_update_machine_info(
    config: pytest.Config, machine_info: dict[str, object]
) -> None:
    """Keep the host name out of saved and committed benchmark reports."""
    _ = config
    machine_info["node"] = "redacted"
