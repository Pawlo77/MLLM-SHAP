"""Explain pipeline types and orchestration.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
"""

from .config import (
    ExplainConfig,
    explain_config_from_yaml,
    explain_config_to_yaml,
)
from .faithfulness import (
    DeletionCurve,
    RemovalMetrics,
    compute_removal_metrics,
    deletion_curve,
    removal_metrics,
)
from .interactions import PairwiseInteractions, explain_interactions
from .manifest import RunManifest, RunManifestPayload, parse_manifest
from .result import ExplainResult, ExplainRunOutput, HierarchicalRunOutput
from .runner import ExplainRunner

__all__ = [
    "DeletionCurve",
    "ExplainConfig",
    "ExplainResult",
    "ExplainRunOutput",
    "ExplainRunner",
    "HierarchicalRunOutput",
    "PairwiseInteractions",
    "RemovalMetrics",
    "RunManifest",
    "RunManifestPayload",
    "compute_removal_metrics",
    "deletion_curve",
    "explain_config_from_yaml",
    "explain_config_to_yaml",
    "explain_interactions",
    "parse_manifest",
    "removal_metrics",
]
