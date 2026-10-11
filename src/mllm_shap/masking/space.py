"""Mask-space utilities for explainable feature indexing.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_space.py]]
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ..domain.coalition import CoalitionMask


@dataclass(frozen=True, slots=True)
class MaskSpace:
    """Describe explainable feature positions inside a full coalition mask."""

    explainable_mask: tuple[bool, ...]
    """Boolean flags selecting explainable feature positions from the full mask."""

    target_length: int
    """Total length of the full coalition mask including fixed positions."""

    def __post_init__(self) -> None:
        """Validate the target length and explainable positions."""
        if self.target_length <= 0:
            msg = "target_length must be positive"
            raise ValueError(msg)
        if len(self.explainable_mask) > self.target_length:
            msg = "explainable_mask cannot exceed target_length"
            raise ValueError(msg)
        if not any(self.explainable_mask):
            msg = "explainable_mask must include at least one explainable position"
            raise ValueError(msg)

    @property
    def n_features(self) -> int:
        """Return the number of explainable features."""
        return sum(self.explainable_mask)

    def materialize(self, split: CoalitionMask | Sequence[bool]) -> tuple[bool, ...]:
        """Project a split over explainable positions back to the full mask."""
        present = (
            split.present
            if isinstance(split, CoalitionMask)
            else tuple(bool(value) for value in split)
        )
        if len(present) != self.n_features:
            msg = (
                f"split length {len(present)} does not match feature count "
                f"{self.n_features}"
            )
            raise ValueError(msg)

        prepared = [True] * self.target_length
        split_index = 0
        for position, is_explainable in enumerate(self.explainable_mask):
            if is_explainable:
                prepared[position] = present[split_index]
                split_index += 1
        return tuple(prepared)
