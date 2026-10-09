"""Estimand labels for attribution outputs and archive manifests.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_estimands.py]]
"""

from enum import StrEnum
from typing import Literal

EstimandWire = Literal["shapley", "banzhaf"]
"""Wire values used in manifests and configuration payloads."""


class Estimand(StrEnum):
    """Supported cooperative-game value formulations."""

    SHAPLEY = "shapley"
    """Shapley value with coalition weights k!(n-k-1)!/n!."""

    BANZHAF = "banzhaf"
    """Banzhaf index with uniform coalition weights 1/2^(n-1)."""


def estimand_to_wire(estimand: Estimand) -> EstimandWire:
    """Serialize an estimand enum member to its wire value."""
    match estimand:
        case Estimand.SHAPLEY:
            return "shapley"
        case Estimand.BANZHAF:
            return "banzhaf"
        case _ as unreachable:
            msg = f"unsupported estimand: {unreachable!r}"
            raise AssertionError(msg)
