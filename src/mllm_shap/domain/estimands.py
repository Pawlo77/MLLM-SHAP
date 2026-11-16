"""Estimand labels for attribution outputs and archive manifests.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_estimands.py]]
"""

from enum import StrEnum
from typing import Literal

EstimandWire = Literal["shapley", "banzhaf", "beta_shapley", "weighted_banzhaf"]
"""Wire values used in manifests and configuration payloads."""


class Estimand(StrEnum):
    """Supported cooperative-game value formulations."""

    SHAPLEY = "shapley"
    """Shapley value with coalition weights k!(n-k-1)!/n!."""

    BANZHAF = "banzhaf"
    """Banzhaf index with uniform coalition weights 1/2^(n-1)."""

    BETA_SHAPLEY = "beta_shapley"
    """Beta Shapley semivalue; weights set by ``alpha`` and ``beta``."""

    WEIGHTED_BANZHAF = "weighted_banzhaf"
    """Weighted Banzhaf semivalue; each other player is present with probability p."""


def estimand_to_wire(estimand: Estimand) -> EstimandWire:
    """Serialize an estimand enum member to its wire value."""
    match estimand:
        case Estimand.SHAPLEY:
            return "shapley"
        case Estimand.BANZHAF:
            return "banzhaf"
        case Estimand.BETA_SHAPLEY:
            return "beta_shapley"
        case Estimand.WEIGHTED_BANZHAF:
            return "weighted_banzhaf"
        case _ as unreachable:
            msg = f"unsupported estimand: {unreachable!r}"
            raise AssertionError(msg)
