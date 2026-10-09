"""Plugin group identifiers for entry-point loading.

@domain plugins
@see [[knowledge/architecture/plugins]]
@tested-by [[tests/plugins/test_groups.py]]
"""

from enum import StrEnum


class PluginGroup(StrEnum):
    """Entry-point groups registered in packaging metadata."""

    ESTIMATORS = "nlp_shap.estimators"
    """Coalition-sampling estimator plugins."""

    ESTIMANDS = "nlp_shap.estimands"
    """Coalition payoff aggregation plugins."""

    VALUE_FNS = "nlp_shap.value_fns"
    """Utility scoring plugins for coalition outputs."""

    NORMALIZERS = "nlp_shap.normalizers"
    """Post-aggregation presentation normalizer plugins."""

    BACKENDS = "nlp_shap.backends"
    """Model inference backend plugins."""

    PARTITIONS = "nlp_shap.partitions"
    """Player-partitioning plugins for conversation snapshots."""

    ABSENCE_POLICIES = "nlp_shap.absence_policies"
    """Absence-policy plugins for rendering masked snapshots."""

    RENDERERS = "nlp_shap.renderers"
    """Attribution visualization renderer plugins."""
