"""Packed bitset encoding and stable hashing for coalition masks.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_codec.py]]
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PackedMask:
    """Packed bit representation of one boolean mask."""

    words: bytes
    """Little-endian packed bytes encoding the boolean mask."""

    n_bits: int
    """Original number of bits (mask length) before packing."""


class MaskCodec:
    """Encode and decode boolean masks to packed bytes."""

    @staticmethod
    def normalize(mask: Sequence[bool] | NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Normalize incoming mask to a 1D boolean array."""
        array = np.asarray(mask, dtype=np.bool_)
        if array.ndim == 0:
            msg = "mask must contain at least one value"
            raise ValueError(msg)
        if array.ndim > 1:
            if array.shape[0] != 1:
                msg = "mask must be 1D or have exactly one row"
                raise ValueError(msg)
            array = array.reshape(-1)
        return array

    @staticmethod
    def pack(mask: Sequence[bool] | NDArray[np.bool_]) -> PackedMask:
        """Pack a mask into little-endian bytes."""
        normalized = MaskCodec.normalize(mask)
        n_bits = int(normalized.size)
        packed = np.packbits(normalized.astype(np.uint8), bitorder="little")
        return PackedMask(words=packed.tobytes(), n_bits=n_bits)

    @staticmethod
    def unpack(packed: PackedMask) -> tuple[bool, ...]:
        """Unpack bytes back to a boolean tuple."""
        raw = np.frombuffer(packed.words, dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="little")[: packed.n_bits]
        return tuple(bool(value) for value in bits)

    @staticmethod
    def hash_mask(mask: Sequence[bool] | NDArray[np.bool_]) -> int:
        """Return a process-independent 64-bit hash over packed mask bytes."""
        packed = MaskCodec.pack(mask)
        digest = hashlib.blake2b(
            packed.words + packed.n_bits.to_bytes(4, "little"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little", signed=False)
