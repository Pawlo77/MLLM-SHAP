"""Archive and reanalyze with real logprobs.

@see [[src/mllm_shap/pipeline/reanalyze.py]]
"""

from pathlib import Path
from typing import Any

import pytest

from mllm_shap.runtime.archive import RunArchive
from tests.practical._harness import (
    PracticalModel,
    Scenario,
    archived,
    scenario_cases,
    summary,
)


@pytest.mark.parametrize("scenario", scenario_cases("archive.reanalyze"))
def test_reanalyze_reproduces_logprob_values_without_generating(
    scenario: Scenario,
    fast_model: PracticalModel,
    practical_record: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Archived logprobs re-score to the same values with zero executions."""
    output = fast_model.explain(scenario, archived(tmp_path))
    again = fast_model.reanalyze(scenario, tmp_path)
    archive = RunArchive.load(tmp_path)
    try:
        rows = list(archive.history_lazy())
    finally:
        archive.close()
    practical_record["numbers"] = {"explain": summary(output)}

    assert again.result.values == pytest.approx(output.result.values, abs=1e-12)
    assert again.result.base_value == pytest.approx(output.result.base_value)
    assert again.metrics.executed == 0
    assert len(rows) == 2 ** len(output.result.values)
    assert all(row.logprobs and max(row.logprobs) <= 0.0 for row in rows)
