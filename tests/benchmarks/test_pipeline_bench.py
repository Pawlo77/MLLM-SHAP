"""End-to-end benchmark of the explain pipeline with the mock backend.

@see [[src/nlp_shap/pipeline/orchestrator.py]]
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role

_CONFIG = ExplainConfig.model_validate({
    "backend": {"kind": "mock", "model_id": "mock"},
    "explanation": {
        "estimator": "exact",
        "value_fn": "tfidf_cosine",
        "absence_policy": "delete",
        "max_inflight": 4,
        "archive": {"path": ""},
    },
})
_SNAPSHOT = ConversationSnapshot.from_turns((
    Turn(
        messages=(
            Message(role=Role.USER, text="one two three four five six seven eight"),
            Message(role=Role.ASSISTANT, text="nine ten"),
        )
    ),
))


@pytest.mark.bench
def test_pipeline_mock_exact_n10(benchmark: BenchmarkFixture) -> None:
    """Exact explain of 10 players (1 024 coalitions) with the mock backend."""
    runner = ExplainRunner(_CONFIG)

    output = benchmark(runner.explain_sync, _SNAPSHOT)

    assert output.metrics.requested == 1024
    assert sum(output.result.values) == pytest.approx(
        output.result.grand_value - output.result.base_value, abs=1e-9
    )
