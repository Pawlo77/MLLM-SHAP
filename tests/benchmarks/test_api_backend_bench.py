"""Performance benchmarks for the API backend.

@see [[src/nlp_shap/backends/api/payload.py]]
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from nlp_shap.backends.api.payload import build_chat_payload, payload_cache_key
from nlp_shap.domain.config import GenerationConfig


@pytest.mark.bench
def test_payload_cache_key_hashing(benchmark: BenchmarkFixture) -> None:
    """Hashing 10k chat payloads for per-request dedup keys."""
    payload = build_chat_payload(
        "remote-model",
        [{"role": "user", "content": "token attribution smoke"}],
        GenerationConfig(max_new_tokens=32, temperature=0.0, top_k=1),
    )

    keys = benchmark(lambda: [payload_cache_key(payload) for _ in range(10_000)])

    assert len(keys[0]) == 64
