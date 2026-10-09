"""Deep tier: KV prefix cache, dedup and archive, and CLI parity on the 3B model.

@see [[src/nlp_shap/runtime/kv_cache.py]]
"""

import time
from pathlib import Path
from typing import Any

import pytest

from nlp_shap.runtime.archive import RunArchive
from tests.practical._harness import (
    ModelPool,
    PracticalModel,
    Scenario,
    archived,
    cli_explain,
    scenario_cases,
    summary,
)


@pytest.mark.parametrize("scenario", scenario_cases("deep.runtime.kv_cache"))
def test_prefix_cache_is_faster_and_equal(
    scenario: Scenario, deep_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Exact over six messages: cache on equals off within float16, and is faster."""
    seconds, outputs = [], []
    for variant in ({}, scenario.variants[0]):  # on first: warm-up favours off
        started = time.perf_counter()
        outputs.append(deep_model.explain(scenario, variant, deep_model.backend))
        seconds.append(time.perf_counter() - started)
    on, off = outputs
    gap = max(
        abs(a - b) for a, b in zip(on.result.values, off.result.values, strict=True)
    )
    speedup = seconds[1] / seconds[0]
    practical_record["numbers"] = {"on": summary(on), "off": summary(off)}
    practical_record["properties"] = {
        "max_gap": gap,
        "seconds_on_off": [round(s, 1) for s in seconds],
        "speedup": round(speedup, 2),
        "kv_cache_hits": on.metrics.kv_cache_hits,
        "kv_cache_reused_tokens": on.metrics.kv_cache_reused_tokens,
    }

    assert on.metrics.kv_cache_hits > 0
    assert gap <= scenario.expect["max_gap"]
    assert speedup >= scenario.expect["min_speedup"]


@pytest.mark.parametrize("scenario", scenario_cases("deep.runtime.dedup_archive"))
def test_archive_reanalyzes_with_logprobs(
    scenario: Scenario,
    deep_model: PracticalModel,
    practical_record: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Reanalyze reproduces the run and re-scores archived logprobs, executing 0."""
    output = deep_model.explain(scenario, archived(tmp_path))
    again = deep_model.reanalyze(scenario, tmp_path)
    rescored = deep_model.reanalyze(scenario, tmp_path, scenario.variants[0])
    direct = deep_model.explain(scenario, scenario.variants[0])
    archive = RunArchive.load(tmp_path)
    try:
        rows = list(archive.history_lazy())
    finally:
        archive.close()
    metrics = output.metrics
    practical_record["numbers"] = {"explain": summary(output)}
    practical_record["properties"] = {
        "requested": metrics.requested,
        "deduplicated": metrics.deduplicated,
        "dedup_ratio": round(metrics.deduplicated / metrics.requested, 3),
        "archive_bytes": sum(
            f.stat().st_size for f in tmp_path.rglob("*") if f.is_file()
        ),
        "archived_rows": len(rows),
    }

    assert again.result.values == pytest.approx(output.result.values, abs=1e-12)
    assert again.metrics.executed == rescored.metrics.executed == 0
    assert rescored.result.values == pytest.approx(direct.result.values, abs=1e-12)
    assert all(row.logprobs and max(row.logprobs) <= 0.0 for row in rows)


@pytest.mark.parametrize("scenario", scenario_cases("deep.cli.parity"))
def test_cli_matches_in_process_values(
    scenario: Scenario,
    deep_model: PracticalModel,
    practical_models: ModelPool,
    practical_record: dict[str, Any],
    tmp_path: Path,
) -> None:
    """``nlp-shap explain`` at budget 128 equals the in-process run for the seed."""
    in_process = deep_model.explain(scenario)
    practical_models.release("deep")  # the subprocess loads its own copy
    started = time.perf_counter()
    payload = cli_explain(scenario, deep_model, tmp_path)
    practical_record["numbers"] = {"in_process": summary(in_process)}
    practical_record["properties"] = {
        "cli_values": payload["values"],
        "cli_seconds": round(time.perf_counter() - started, 1),
    }

    assert payload["values"] == pytest.approx(in_process.result.values, abs=1e-12)
