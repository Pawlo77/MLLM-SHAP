"""Value functions and normalizers on real generations.

@see [[src/mllm_shap/value/logprob.py]]
@see [[src/mllm_shap/value/wer.py]]
@see [[src/mllm_shap/pipeline/faithfulness.py]]
"""

import math
from pathlib import Path
from typing import Any

import pytest

from mllm_shap import DeletionCurve, deletion_curve
from tests.practical._harness import (
    LOOP,
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
    scenario_cases("value.tfidf_cosine", "value.embedding_cosine"),
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


@pytest.mark.parametrize("scenario", scenario_cases("value.self_logprob"))
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


@pytest.mark.parametrize("scenario", scenario_cases("value.reference_wer"))
def test_shapley_ranking_beats_its_reverse_on_a_deletion_curve(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Removing the top Shapley words first destroys the answer sooner."""
    exact = default_model.explain(SCENARIOS[scenario.expect["explained_by"]])
    values = exact.result.values
    config = scenario.config(default_model.spec)

    def curve(ranking: list[float]) -> DeletionCurve:
        return LOOP.run_until_complete(
            deletion_curve(
                config, scenario.snapshot, ranking, backend=default_model.cached
            )
        )

    ranked, reverse = curve(values), curve([-value for value in values])
    practical_record["numbers"] = {
        "ranked": list(ranked.values),
        "reverse": list(reverse.values),
    }
    practical_record["properties"] = {"area": ranked.area, "reverse_area": reverse.area}

    assert ranked.values[0] == pytest.approx(1.0)  # the full answer matches itself
    assert ranked.area < reverse.area


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


@pytest.mark.parametrize(
    "scenario",
    scenario_cases(
        "value.logprob",
        "value.tf_logprob",
        "value.tf_jsd",
        "value.contrastive",
        "value.bertscore",
        "value.nli_logodds",
        "value.semascore",
        "value.judge_grade",
        "value.speech_bertscore",
        "value.transcribed",
    ),
)
def test_new_value_functions_give_finite_attributions(
    scenario: Scenario, default_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Each value function runs end to end on a real model and keeps its sign."""
    if "skip" in scenario.expect:
        pytest.skip(str(scenario.expect["skip"]))
    output = default_model.explain(scenario)
    practical_record["numbers"] = summary(output)

    assert all(math.isfinite(v) for v in output.result.values)
    if scenario.expect.get("sign") == -1:
        assert output.result.base_value <= 0.0
        assert output.result.grand_value <= 0.0
    if "top_player" in scenario.expect:
        assert scenario.expect["top_player"] in top_players(
            labels(output), output.result.values
        )
