"""The command line on a real model matches the in-process pipeline.

@see [[src/mllm_shap/cli.py]]
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.practical._harness import (
    PracticalModel,
    Scenario,
    cli_explain,
    run_cli,
    scenario_cases,
)


@pytest.mark.parametrize("scenario", scenario_cases("cli.explain"))
def test_cli_explain_matches_in_process_values(
    scenario: Scenario,
    fast_model: PracticalModel,
    practical_record: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Subprocess values equal the in-process run; the PNG is written."""
    payload = cli_explain(scenario, fast_model, tmp_path)
    in_process = fast_model.explain(scenario)
    practical_record["numbers"] = {"cli": payload["values"]}

    assert payload["values"] == pytest.approx(in_process.result.values, abs=1e-12)
    assert (tmp_path / "figure.png").stat().st_size > 0


@pytest.mark.parametrize("scenario", scenario_cases("cli.reanalyze"))
def test_cli_reanalyze_matches_cli_explain(
    scenario: Scenario,
    fast_model: PracticalModel,
    practical_record: dict[str, Any],
    tmp_path: Path,
) -> None:
    """``mllm-shap reanalyze`` re-scores the CLI's archive to the same values."""
    payload = cli_explain(scenario, fast_model, tmp_path)
    run_cli(
        "reanalyze",
        "--config",
        str(tmp_path / "config.yaml"),
        "--archive",
        str(tmp_path / "run"),
        "--out",
        str(tmp_path / "again.json"),
    )
    again = json.loads((tmp_path / "again.json").read_text("utf-8"))
    practical_record["numbers"] = {
        "explain": payload["values"],
        "again": again["values"],
    }

    assert again["values"] == pytest.approx(payload["values"], abs=1e-12)
    assert again["metrics"]["executed"] == 0
