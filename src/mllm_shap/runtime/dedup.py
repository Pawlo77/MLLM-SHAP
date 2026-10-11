"""Coalition deduplication keys and registry.

@domain runtime
@see [[knowledge/architecture/runtime]]
@tested-by [[tests/runtime/test_dedup.py]]
"""

import hashlib
import struct
from dataclasses import dataclass

from ..domain.config import DedupConfig, GenerationConfig
from ..masking.codec import MaskCodec, PackedMask


def dedup_enabled(config: DedupConfig, generation: GenerationConfig) -> bool:
    """Return whether coalition deduplication should be active."""
    match config.enabled:
        case "on":
            return True
        case "off":
            return False
        case "auto":
            return generation.temperature == 0.0
        case _ as unreachable:
            msg = f"unsupported dedup mode: {unreachable!r}"
            raise AssertionError(msg)


def _hash_packed_mask(packed: PackedMask) -> bytes:
    """Return stable bytes for a packed coalition mask."""
    return struct.pack("<I", packed.n_bits) + packed.words


def build_coalition_key(
    snapshot_id: str,
    player_ids: tuple[str, ...],
    mask_present: tuple[bool, ...],
    absence_policy: str,
    model_id: str,
    generation: GenerationConfig,
) -> str:
    """Build a stable SHA256 coalition key for deduplication."""
    packed = MaskCodec.pack(mask_present)
    hasher = hashlib.sha256()
    hasher.update(snapshot_id.encode("utf-8"))
    hasher.update(b"\x1e")
    for player_id in player_ids:
        hasher.update(player_id.encode("utf-8"))
        hasher.update(b"\x1f")
    hasher.update(_hash_packed_mask(packed))
    hasher.update(absence_policy.encode("utf-8"))
    hasher.update(model_id.encode("utf-8"))
    hasher.update(
        struct.pack(
            "<ifi",
            generation.max_new_tokens,
            generation.temperature,
            generation.top_k,
        )
    )
    return hasher.hexdigest()


@dataclass
class CoalitionDedupRegistry:
    """Track coalition keys that already required backend execution."""

    def __init__(self) -> None:
        """Start with no observed coalition keys."""
        self._seen: set[str] = set()

    def observe(self, coalition_key: str) -> bool:
        """Record a coalition key and return whether it is newly observed."""
        is_new = coalition_key not in self._seen
        if is_new:
            self._seen.add(coalition_key)
        return is_new

    def contains(self, coalition_key: str) -> bool:
        """Return whether a coalition key was already observed."""
        return coalition_key in self._seen

    def __len__(self) -> int:
        """Return the number of unique coalition keys observed."""
        return len(self._seen)
