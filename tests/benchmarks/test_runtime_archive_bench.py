"""Micro-benchmarks for archive persistence.

@see [[src/nlp_shap/runtime/archive.py]]
"""

import itertools
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from nlp_shap.domain.estimands import Estimand
from nlp_shap.masking.codec import MaskCodec
from nlp_shap.pipeline.manifest import RunManifest
from nlp_shap.runtime.archive import CoalitionRecordDraft, RunArchive

_MANIFEST = RunManifest(estimand=Estimand.SHAPLEY, run_id="bench-archive")
_PACKED = MaskCodec.pack((True, False, True, False))


@pytest.mark.bench
def test_archive_round_trip_10k(tmp_path: Path, benchmark: BenchmarkFixture) -> None:
    """Appending and lazily reading 10k coalition rows."""
    counter = itertools.count()

    def round_trip() -> int:
        root = tmp_path / f"bench-run-{next(counter)}"
        with RunArchive.open(root, _MANIFEST, flush_every=250) as archive:
            for index in range(10_000):
                archive.append(
                    CoalitionRecordDraft(
                        snapshot_id="bench-snapshot",
                        coalition_key=f"key-{index % 128}",
                        mask=_PACKED,
                        absence_policy="pad",
                        model_id="mock",
                        generation_text=f"generation-{index}",
                        elapsed_ms=1.0,
                        cache_hit=index % 2 == 0,
                    )
                )
        with RunArchive.load(root) as archive:
            return sum(1 for _ in archive.history_lazy())

    assert benchmark(round_trip) == 10_000
