"""Rendered labels of a real multi-speaker explanation.

@see [[src/nlp_shap/viz/render.py]]
"""

from typing import Any

import pytest

from nlp_shap.viz import render_attribution
from tests.practical._harness import PracticalModel, Scenario, labels, scenario_cases


@pytest.mark.parametrize("scenario", scenario_cases("viz.render"))
def test_bar_labels_are_span_labels_with_speakers(
    scenario: Scenario, fast_model: PracticalModel, practical_record: dict[str, Any]
) -> None:
    """Every bar label is a player's display label and names its speaker."""
    pytest.importorskip("matplotlib")
    output = fast_model.explain(scenario)
    assert output.player_set is not None
    figure = render_attribution(
        output, scenario.snapshot, output.player_set, renderer="token_bar"
    )
    shown = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
    expected = [span.display_label for span in output.player_set.spans]
    practical_record["properties"] = {"labels": shown}

    assert sorted(shown) == sorted(expected) == sorted(labels(output))
    for label, speaker in zip(expected, scenario.expect["speakers"], strict=True):
        assert f"/{speaker} " in label
