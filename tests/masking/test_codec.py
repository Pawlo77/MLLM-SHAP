"""Tests for packed mask encoding and hashing.

@see [[src/mllm_shap/masking/codec.py]]
"""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from mllm_shap.masking.codec import MaskCodec, PackedMask

LEGACY_TOY_MASKS: tuple[tuple[bool, ...], ...] = (
    (True, False, True, True, False, False, True),
    (True, False, True, False),
)


def test_pack_unpack_roundtrip() -> None:
    """Packed masks round-trip to the original boolean sequence."""
    mask = LEGACY_TOY_MASKS[0]
    packed = MaskCodec.pack(mask)
    restored = MaskCodec.unpack(packed)
    assert restored == mask


@pytest.mark.parametrize("mask", LEGACY_TOY_MASKS)
def test_hash_stable_across_shapes(mask: tuple[bool, ...]) -> None:
    """Row-shaped and flat masks hash to the same packed value."""
    row = np.asarray([mask], dtype=np.bool_)
    flat = np.asarray(mask, dtype=np.bool_)
    assert MaskCodec.hash_mask(row) == MaskCodec.hash_mask(flat)


@pytest.mark.parametrize("mask", LEGACY_TOY_MASKS)
def test_hash_matches_legacy_pack_bytes(mask: tuple[bool, ...]) -> None:
    """Packed bytes match the 0.x mask codec; the hash digests those bytes."""
    packed = MaskCodec.pack(mask)
    legacy_packed = _legacy_pack(mask)
    assert packed.words == legacy_packed.words
    assert packed.n_bits == legacy_packed.n_bits
    digest = hashlib.blake2b(
        packed.words + packed.n_bits.to_bytes(4, "little"), digest_size=8
    ).digest()
    assert MaskCodec.hash_mask(mask) == int.from_bytes(digest, "little")


def test_normalize_rejects_multi_row_masks() -> None:
    """Only single-row 2D masks are accepted."""
    invalid = np.asarray([[True, False], [False, True]], dtype=np.bool_)
    with pytest.raises(ValueError, match="exactly one row"):
        MaskCodec.normalize(invalid)


def _legacy_pack(mask: tuple[bool, ...]) -> PackedMask:
    """Reference packer mirroring legacy MLLM-Shap ``MaskCodec.pack``."""
    bits = np.asarray(mask, dtype=np.uint8)
    packed = np.packbits(bits, bitorder="little")
    return PackedMask(words=packed.tobytes(), n_bits=len(mask))


def test_hash_mask_is_stable_across_processes() -> None:
    code = (
        "from mllm_shap.masking.codec import MaskCodec;"
        "print(MaskCodec.hash_mask((True, False, True, True)))"
    )
    outs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        ).stdout.strip()
        for seed in ("1", "2")
    }
    assert len(outs) == 1
