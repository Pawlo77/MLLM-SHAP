"""Token hashing helpers for TF-IDF value functions.

@domain value
@see [[knowledge/architecture/value]]
"""

import hashlib
from collections.abc import Sequence

import numpy as np

from ..domain.generation import GenerationRecord


class TokenHasher:
    """Assign stable integer ids to multimodal token rows."""

    def __init__(self) -> None:
        self._token_map: dict[bytes, int] = {}
        self._token_counter = 0

    def hash_rows(self, rows: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
        """Return stable integer ids for each token row."""
        return tuple(self._hash_row(row) for row in rows)

    def document_token_ids(self, generation: GenerationRecord) -> tuple[int, ...]:
        """Return hashed token ids for text and audio rows."""
        text_ids = self.hash_rows(generation.text_token_rows)
        audio_ids = self.hash_rows(generation.audio_token_rows)
        return text_ids + audio_ids

    def _hash_row(self, row: tuple[int, ...]) -> int:
        array = np.asarray(row, dtype=np.int64)
        digest = hashlib.sha256()
        digest.update(array.tobytes())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(str(array.dtype).encode())
        key = digest.digest()
        if key not in self._token_map:
            self._token_map[key] = self._token_counter
            self._token_counter += 1
        return self._token_map[key]
