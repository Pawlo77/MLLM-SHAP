"""Tests for plugin option lookup.

@see [[src/mllm_shap/plugins/options.py]]
"""

from mllm_shap.plugins import PluginGroup, default_registry
from mllm_shap.plugins.options import non_default_options

POLICIES = PluginGroup.ABSENCE_POLICIES
"""Group of the plugins whose options the coalition key hashes."""


def test_non_default_options_drop_values_equal_to_the_default() -> None:
    registry = default_registry()
    assert non_default_options(registry, {}, POLICIES, "pad") == {}
    explicit = {"pad": {"placeholder": "[MASK]"}}
    assert non_default_options(registry, explicit, POLICIES, "pad") == {}
    custom = {"absence_policies/pad": {"placeholder": "<x>"}}
    assert non_default_options(registry, custom, POLICIES, "pad") == {
        "placeholder": "<x>"
    }
