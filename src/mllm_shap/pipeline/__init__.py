"""Explain pipeline types and orchestration.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
"""

from .config import (
    ExplainConfig,
    explain_config_from_yaml,
    explain_config_to_yaml,
)
from .faithfulness import DeletionCurve, deletion_curve
from .manifest import RunManifest, RunManifestPayload, parse_manifest
from .result import ExplainResult, ExplainRunOutput
from .runner import ExplainRunner

__all__ = [
    "DeletionCurve",
    "ExplainConfig",
    "ExplainResult",
    "ExplainRunOutput",
    "ExplainRunner",
    "RunManifest",
    "RunManifestPayload",
    "deletion_curve",
    "explain_config_from_yaml",
    "explain_config_to_yaml",
    "parse_manifest",
]
