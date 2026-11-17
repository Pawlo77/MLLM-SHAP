"""Cost accounting: model work per run from the backend's counters.

@see [[src/mllm_shap/runtime/metrics.py]]
"""

import asyncio

import pytest

from mllm_shap import ExplainConfig, ExplainRunner
from mllm_shap.backends.mock import MockBackend
from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.protocols.backend import ReportsCost
from mllm_shap.runtime.metrics import CostCounters


def _snapshot(words: int = 4) -> ConversationSnapshot:
    text = " ".join(f"w{i}" for i in range(words))
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _config(**explanation: object) -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "mock"},
        "generation": {"max_new_tokens": 4},
        "explanation": {
            "absence_policy": "pad",
            "archive": {"path": ""},
            **explanation,
        },
    })


def test_counters_subtract_to_the_work_done_between_two_reads() -> None:
    after = CostCounters(5, 100, 40, 2.5)
    before = CostCounters(2, 30, 10, 1.0)
    assert after - before == CostCounters(3, 70, 30, 1.5)


def test_mock_backend_counts_every_sequence_and_token() -> None:
    backend = MockBackend()
    assert isinstance(backend, ReportsCost)
    record = asyncio.run(
        backend.generate(_snapshot(4), GenerationConfig(max_new_tokens=4))
    )
    counters = backend.cost_counters
    assert counters.forward_sequences == 1
    assert counters.generated_tokens == len(record.text.split())
    assert counters.prompt_tokens > 0
    assert counters.device_seconds == 0.0


def test_a_run_reports_the_sum_of_the_backend_calls() -> None:
    backend = MockBackend()
    output = ExplainRunner(_config(estimator="exact"), backend=backend).explain_sync(
        _snapshot(3)
    )
    perf = output.perf
    assert perf is not None
    total = backend.cost_counters
    assert perf.forward_sequences == total.forward_sequences
    assert perf.prompt_tokens == total.prompt_tokens
    assert perf.generated_tokens == total.generated_tokens
    # The base answer and every coalition except the seeded grand one are generated.
    assert perf.forward_sequences == output.metrics.executed + 1
    assert perf.generated_tokens == pytest.approx(4 * perf.forward_sequences)


def test_a_second_run_on_the_same_backend_reports_only_its_own_work() -> None:
    backend = MockBackend()
    config = _config(estimator="exact")
    first = ExplainRunner(config, backend=backend).explain_sync(_snapshot(3)).perf
    second = ExplainRunner(config, backend=backend).explain_sync(_snapshot(3)).perf
    assert first is not None
    assert second is not None
    assert second.forward_sequences == first.forward_sequences
    assert backend.cost_counters.forward_sequences == 2 * first.forward_sequences


def test_the_payload_carries_the_cost_fields() -> None:
    output = ExplainRunner(_config(estimator="exact")).explain_sync(_snapshot(3))
    perf = output.to_dict()["perf"]
    assert perf is not None
    assert {
        "forward_sequences",
        "prompt_tokens",
        "generated_tokens",
        "device_seconds",
    } <= set(perf)
