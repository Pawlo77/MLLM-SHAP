"""Micro-benchmarks for mask pack and unpack throughput.

@see [[src/mllm_shap/masking/codec.py]]
"""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from mllm_shap.masking.codec import MaskCodec

_MASKS = [tuple(bit == "1" for bit in format(index, "08b")) for index in range(256)]


def _pack_unpack_10k() -> int:
    restored_bits = 0
    for index in range(10_000):
        restored = MaskCodec.unpack(MaskCodec.pack(_MASKS[index % len(_MASKS)]))
        restored_bits += len(restored)
    return restored_bits


@pytest.mark.bench
def test_mask_pack_unpack_10k(benchmark: BenchmarkFixture) -> None:
    """Packing and unpacking 10k eight-player masks."""
    assert benchmark(_pack_unpack_10k) == 80_000
