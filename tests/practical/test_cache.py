"""Generation and verdict cache, without a model.

@see [[src/mllm_shap/domain/generation.py]]
"""

import asyncio
from pathlib import Path

from mllm_shap.domain.config import GenerationConfig
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.generation import GenerationRecord
from tests.practical._cache import CachedBackend, PracticalCache

RECORD = GenerationRecord(
    text="Paris.",
    text_token_rows=((1, 2),),
    logprobs=(-0.5, -0.25),
    embedding=(0.5, 1.0),
    contextual_embedding=(0.25,),
)


class CountingBackend:
    """Return ``RECORD`` and count the calls."""

    model_id = "demo/model"
    calls = 0

    async def generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        self.calls += 1
        return RECORD


def _snapshot(text: str) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text=text),)),
    ))


def _cached(path: Path, inner: CountingBackend) -> CachedBackend:
    identity = {"model": "demo/model", "revision": "abc", "device": "mps"}
    return CachedBackend(inner, PracticalCache(path), identity)


def test_second_identical_generation_is_served_from_disk(tmp_path: Path) -> None:
    """A new cache on the same file returns the stored record unchanged."""
    inner = CountingBackend()
    path = tmp_path / "cache.sqlite"
    generation = GenerationConfig(max_new_tokens=8)
    first = asyncio.run(_cached(path, inner).generate(_snapshot("a b"), generation))
    again = _cached(path, inner)
    second = asyncio.run(again.generate(_snapshot("a b"), generation))

    assert first == second == RECORD
    assert inner.calls == 1
    assert (again.hits, again.misses) == (1, 0)


def test_other_prompt_or_generation_config_misses(tmp_path: Path) -> None:
    """The key covers the rendered chat and the generation config."""
    inner = CountingBackend()
    backend = _cached(tmp_path / "cache.sqlite", inner)

    async def run() -> None:
        await backend.generate(_snapshot("a b"), GenerationConfig(max_new_tokens=8))
        await backend.generate(_snapshot("a c"), GenerationConfig(max_new_tokens=8))
        await backend.generate(_snapshot("a b"), GenerationConfig(max_new_tokens=9))

    asyncio.run(run())
    assert inner.calls == 3
    assert backend.model_id == "demo/model"
    assert backend.returns_logprobs


def test_values_round_trip(tmp_path: Path) -> None:
    """Verdicts and other JSON values come back as stored."""
    cache = PracticalCache(tmp_path / "cache.sqlite")
    cache.put("verdict", "k", {"margins": [1.5, -2.0]})
    assert cache.get("verdict", "k") == {"margins": [1.5, -2.0]}
    assert cache.get("verdict", "missing") is None
