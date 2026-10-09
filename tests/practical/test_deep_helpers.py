"""Deep-tier helpers without a model: focused partition and error tables.

@see [[src/nlp_shap/masking/partitions.py]]
"""

import pytest

from nlp_shap import ExplainConfig
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.pipeline.config import BackendConfig
from nlp_shap.plugins import PluginContext, PluginGroup
from tests.practical._deep import (
    REGISTRY,
    error_table,
    non_increasing,
    top1_agreement,
)
from tests.practical._report import render


def test_focus_tokens_keeps_one_message() -> None:
    """Only the chosen message's whitespace tokens become players."""
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="a b"),)),
        Turn(messages=(Message(role=Role.USER, text="c d e"),)),
    ))
    config = ExplainConfig.model_validate({
        "backend": BackendConfig(kind="mock", model_id="m"),
        "explanation": {"plugin_options": {"focus_tokens": {"message_index": 1}}},
    })
    partition = REGISTRY.resolve(
        PluginGroup.PARTITIONS, "focus_tokens", PluginContext(config=config)
    )
    players = partition.partition(snapshot)  # type: ignore[attr-defined]
    assert [s.text for s in players.spans] == ["c", "d", "e"]
    assert {s.message_index for s in players.spans} == {1}


def test_error_table_medians_and_monotonicity() -> None:
    """Normalised L1 per seed and its median, per method and budget."""
    reference = [1.0, -1.0]
    runs = {"mc": {8: [[0.5, -1.0], [1.0, -0.5], [0.0, -1.0]], 32: [[1.0, -1.0]] * 3}}
    table = error_table(runs, reference)
    assert table["mc"][8]["per_seed"] == pytest.approx([0.25, 0.25, 0.5])
    assert table["mc"][8]["median"] == pytest.approx(0.25)
    assert table["mc"][32]["median"] == 0.0
    assert non_increasing([0.3, 0.3, 0.1])
    assert not non_increasing([0.1, 0.2])


def test_top1_agreement_against_the_reference() -> None:
    """Share of seeds whose largest value is the reference's top player."""
    assert top1_agreement([[0.9, 0.1], [0.2, 0.3], [0.5, 0.4]], [1.0, 0.0]) == 2 / 3


def test_report_renders_scenarios_and_error_tables() -> None:
    """One row per scenario, one per method and budget, and the run line."""
    cell = {"median": 0.1, "per_seed": [0.1, 0.2]}
    payload = {
        "tier": "deep",
        "device": "mps",
        "wall_seconds": 3.0,
        "peak_vram_gb": 6.5,
        "cache": {},
        "scenarios": {
            "n": {
                "feature": "deep.x",
                "outcome": "passed",
                "seconds": 1.0,
                "driver_gb": 6.5,
                "tables": {"error_vs_budget": {"mc": {"32": cell}}},
            }
        },
    }
    lines = render(payload).splitlines()
    assert "| deep.x | passed | 1.0 | 6.5 |" in lines
    assert "| deep.x | mc | 32 | 0.1 | [0.1, 0.2] |" in lines
    assert lines[-1].startswith("tier deep, device mps")
