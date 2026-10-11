"""Estimand aggregation plugins.

@domain estimation
@see [[knowledge/architecture/estimation]]
"""

from .banzhaf import BanzhafAggregator
from .shapley import ShapleyAggregator

__all__ = ["BanzhafAggregator", "ShapleyAggregator"]
