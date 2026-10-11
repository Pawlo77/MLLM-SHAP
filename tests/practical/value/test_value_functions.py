"""Value functions and normalizers on real generations.

@see [[src/mllm_shap/value/logprob.py]]
"""

import math
from pathlib import Path
from typing import Any

import pytest

from tests.practical._harness import (
    SCENARIOS,
    PracticalModel,
    Scenario,
    archived,
    labels,
    scenario_cases,
    summary,
)
from tests.practical._judge import efficiency_gap, top_by_magnitude, top_players


@pytest.mark.parametrize(
    "scenario",
    scenario_cases(
        "value.tfidf_cosine", "value.embedding_cosine", "value.embedding_euclidean"
    ),
)
def test_similarity_values_agree_on_the_top_player(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Values are finite; TF-IDF ranks France first, embeddings agree with it.

    TF-IDF scores the one-word answer all or nothing, so "capital" and "France"
    tie; an embedding value may break the tie either way.
    """
    output = default_model.explain(scenario)
    values, names = output.result.values, labels(output)
    practical_record["numbers"] = summary(output)

    assert all(math.isfinite(v) for v in values)
    if "top_player" in scenario.expect:
        assert scenario.expect["top_player"] in top_players(names, values)
        return
    reference = default_model.explain(SCENARIOS[scenario.expect["agrees_with"]])
    assert set(top_players(names, values)) <= set(
        top_players(names, reference.result.values)
    )


@pytest.mark.parametrize("scenario", scenario_cases("value.logprob"))
def test_logprob_values_flag_the_planted_contradiction(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Mean token logprobs are at most 0; the contradiction lowers confidence."""
    output = default_model.explain(scenario)
    result = output.result
    planted = scenario.expect["contradiction"]
    practical_record["numbers"] = summary(output)
    practical_record["properties"] = {
        "top_by_magnitude": top_by_magnitude(result.values, 1)[0]
    }

    assert result.base_value <= 0.0
    assert result.grand_value <= 0.0
    assert efficiency_gap(result.values, result.base_value, result.grand_value) < 1e-9
    assert math.copysign(1.0, result.values[planted]) == scenario.expect["sign"]


@pytest.mark.parametrize("scenario", scenario_cases("value.normalizers"))
def test_normalizers_rescore_one_archive(
    scenario: Scenario,
    fast_model: PracticalModel,
    practical_record: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Each normalizer re-scores the archive without generating; order is kept."""
    output = fast_model.explain(scenario, archived(tmp_path))
    identity = fast_model.reanalyze(scenario, tmp_path)
    scaled = {
        v["explanation"]["normalizer"]: fast_model.reanalyze(scenario, tmp_path, v)
        for v in scenario.variants
    }
    practical_record["numbers"] = {
        name: [round(x, 6) for x in run.result.values] for name, run in scaled.items()
    }
    names = labels(output)
    top = top_players(names, output.result.values)

    assert identity.result.values == pytest.approx(output.result.values, abs=1e-12)
    for name, run in {"identity": identity, **scaled}.items():
        assert run.metrics.executed == 0, name
        assert top_players(names, run.result.values) == top, name
    assert sum(abs(v) for v in scaled["abs_sum"].result.values) == pytest.approx(1.0)
    for name in ("power_shift", "min_max"):
        values = scaled[name].result.values
        assert (sum(values), min(values)) == pytest.approx((1.0, 0.0)), name
