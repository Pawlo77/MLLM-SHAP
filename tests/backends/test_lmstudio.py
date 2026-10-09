"""Integration tests for the LM Studio backend.

@see [[src/nlp_shap/backends/lmstudio/__init__.py]]
"""

import asyncio
from pathlib import Path

import pytest

from nlp_shap.backends.lmstudio import LmStudioBackend
from nlp_shap.domain.config import GenerationConfig
from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.errors import BackendUnavailableError
from nlp_shap.pipeline.config import BackendConfig, explain_config_from_yaml
from nlp_shap.pipeline.runner import ExplainRunner
from tests.backends._context import backend_context

pytestmark = pytest.mark.lms


@pytest.fixture(scope="module")
def lmstudio_host() -> str:
    """Return a reachable LM Studio API host or skip the module."""
    lmstudio = pytest.importorskip("lmstudio")
    host = asyncio.run(lmstudio.AsyncClient.find_default_local_api_host())
    if host is None:
        pytest.skip("LM Studio API server is not running")
    if not asyncio.run(lmstudio.AsyncClient.is_valid_api_host(host)):
        pytest.skip(f"LM Studio API host is invalid: {host!r}")
    return host


def _user_snapshot(text: str) -> ConversationSnapshot:
    turn = Turn(messages=(Message(role=Role.USER, text=text),))
    return ConversationSnapshot.from_turns((turn,))


def test_lmstudio_backend_generate_smoke(lmstudio_host: str) -> None:
    """LmStudioBackend returns non-empty text from a loaded local model."""
    backend = LmStudioBackend(
        backend_context(
            BackendConfig(
                kind="lmstudio",
                model_id="qwen2-500m-instruct",
                api_host=lmstudio_host,
            )
        )
    )

    async def run() -> str:
        try:
            record = await backend.generate(
                _user_snapshot("Reply with the word ok."),
                GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=1),
            )
            return record.text
        finally:
            await backend.aclose()

    text = asyncio.run(run())
    assert text


def test_lmstudio_backend_raises_when_host_unreachable() -> None:
    """Invalid API hosts raise BackendUnavailableError without LM Studio calls."""
    backend = LmStudioBackend(
        backend_context(
            BackendConfig(
                kind="lmstudio",
                model_id="qwen2-500m-instruct",
                api_host="127.0.0.1:1",
            )
        )
    )

    async def run() -> None:
        try:
            await backend.generate(
                _user_snapshot("hello"),
                GenerationConfig(max_new_tokens=4, temperature=0.0, top_k=1),
            )
        finally:
            await backend.aclose()

    with pytest.raises(BackendUnavailableError):
        asyncio.run(run())


def test_lmstudio_runner_exact_shapley_smoke(lmstudio_host: str) -> None:
    """ExplainRunner completes a tiny exact run against LM Studio."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "dev_lmstudio.yaml"
    config = explain_config_from_yaml(config_path.read_text(encoding="utf-8"))
    config = config.model_copy(
        update={
            "backend": config.backend.model_copy(update={"api_host": lmstudio_host}),
        }
    )

    output = ExplainRunner(config).explain_sync(_user_snapshot("one two three"))

    assert len(output.result.values) == 3
    assert output.metrics.executed > 0
