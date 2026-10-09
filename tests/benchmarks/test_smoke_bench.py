"""Import-time benchmark for the package.

@see [[src/nlp_shap/__init__.py]]
"""

import subprocess
import sys

import pytest
from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.bench
def test_import_bench(benchmark: BenchmarkFixture) -> None:
    """``import nlp_shap`` in a fresh interpreter (no cached modules)."""

    def import_fresh() -> int:
        return subprocess.run(
            [sys.executable, "-c", "import nlp_shap"], check=False
        ).returncode

    assert benchmark(import_fresh) == 0
