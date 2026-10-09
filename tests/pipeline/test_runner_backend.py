"""Backend injection and event-loop guard of ExplainRunner.

@see [[src/nlp_shap/pipeline/runner.py]]
"""

import asyncio

import pytest

from nlp_shap import ExplainConfig, ExplainRunner
from nlp_shap.backends.mock import MockBackend
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role


class ClosableMock(MockBackend):
    """Mock backend that counts generate and aclose calls."""

    def __init__(self) -> None:
        super().__init__(model_id="stub")
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


def _config() -> ExplainConfig:
    return ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "estimator": "exact",
            "absence_policy": "pad",
            "archive": {"path": ""},
        },
    })


def _snapshot(text: str = "a b c") -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def test_injected_backend_is_reused_and_left_open() -> None:
    backend = ClosableMock()
    runner = ExplainRunner(_config(), backend=backend)

    async def run() -> None:
        await runner.explain(_snapshot())
        await runner.explain(_snapshot("d e"))
        assert backend.closed == 0
        await runner.aclose()

    asyncio.run(run())
    assert backend.closed == 1


def test_injected_backend_matches_built_backend_values() -> None:
    built = ExplainRunner(_config()).explain_sync(_snapshot())
    injected = ExplainRunner(_config(), backend=ClosableMock()).explain_sync(
        _snapshot()
    )
    assert injected.result.values == built.result.values


def test_explain_sync_inside_running_loop_raises() -> None:
    runner = ExplainRunner(_config())

    async def _call_sync_inside_loop() -> None:
        runner.explain_sync(_snapshot())

    with pytest.raises(RuntimeError, match=r"await runner\.explain"):
        asyncio.run(_call_sync_inside_loop())


class LoopBoundMock(ClosableMock):
    """Mock backend that, like an HTTP client, binds to the first event loop."""

    def __init__(self) -> None:
        super().__init__()
        self.loops: set[asyncio.AbstractEventLoop] = set()
        self._lock: asyncio.Lock | None = None

    async def generate(self, snapshot, generation):  # type: ignore[no-untyped-def]  # mirrors MockBackend
        loop = asyncio.get_running_loop()
        self.loops.add(loop)
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            return await super().generate(snapshot, generation)


def test_explain_sync_reuses_one_loop_for_injected_backend() -> None:
    backend = LoopBoundMock()
    runner = ExplainRunner(_config(), backend=backend)
    first = runner.explain_sync(_snapshot())
    second = runner.explain_sync(_snapshot("d e"))
    runner.close()

    assert len(backend.loops) == 1
    assert first.result.values
    assert second.result.values
    assert backend.closed == 1
    (loop,) = backend.loops
    assert loop.is_closed()


def test_runner_context_manager_closes_backend_and_loop() -> None:
    backend = LoopBoundMock()
    with ExplainRunner(_config(), backend=backend) as runner:
        runner.explain_sync(_snapshot())
    assert backend.closed == 1
    (loop,) = backend.loops
    assert loop.is_closed()


def test_async_context_manager_closes_backend_used_by_explain_sync() -> None:
    backend = LoopBoundMock()
    runner = ExplainRunner(_config(), backend=backend)
    runner.explain_sync(_snapshot())

    async def close_from_another_loop() -> None:
        async with runner:
            pass

    asyncio.run(close_from_another_loop())
    assert backend.closed == 1
    (loop,) = backend.loops
    assert loop.is_closed()


def test_run_id_hashes_injected_backend_model_id() -> None:
    first = ExplainRunner(_config(), backend=MockBackend(model_id="model-a"))
    second = ExplainRunner(_config(), backend=MockBackend(model_id="model-b"))
    assert (
        first.explain_sync(_snapshot()).run_id
        != second.explain_sync(_snapshot()).run_id
    )
    first.close()
    second.close()
