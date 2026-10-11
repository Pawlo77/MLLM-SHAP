"""Plugin group identifiers for entry-point loading.

@domain plugins
@see [[knowledge/architecture/plugins]]
@tested-by [[tests/plugins/test_groups.py]]
"""

from enum import StrEnum


class PluginGroup(StrEnum):
    """Entry-point groups registered in packaging metadata."""

    ESTIMATORS = "mllm_shap.estimators"
    """Coalition-sampling estimator plugins."""

    ESTIMANDS = "mllm_shap.estimands"
    """Coalition payoff aggregation plugins."""

    VALUE_FNS = "mllm_shap.value_fns"
    """Utility scoring plugins for coalition outputs."""

    NORMALIZERS = "mllm_shap.normalizers"
    """Post-aggregation presentation normalizer plugins."""

    BACKENDS = "mllm_shap.backends"
    """Model inference backend plugins."""

    PARTITIONS = "mllm_shap.partitions"
    """Player-partitioning plugins for conversation snapshots."""

    ABSENCE_POLICIES = "mllm_shap.absence_policies"
    """Absence-policy plugins for rendering masked snapshots."""

    RENDERERS = "mllm_shap.renderers"
    """Attribution visualization renderer plugins."""
