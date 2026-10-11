"""Every built-in plugin has a practical scenario and every scenario a test.

@see [[src/mllm_shap/plugins/builtin.py]]
"""

from pathlib import Path

import pytest

from mllm_shap.plugins import PluginGroup
from mllm_shap.plugins.builtin import BUILTIN_PLUGINS
from tests.practical._harness import MODELS, SCENARIOS

HERE = Path(__file__).parent

FIELDS = {
    PluginGroup.ESTIMATORS: "estimator",
    PluginGroup.ESTIMANDS: "estimand",
    PluginGroup.VALUE_FNS: "value_fn",
    PluginGroup.NORMALIZERS: "normalizer",
    PluginGroup.ABSENCE_POLICIES: "absence_policy",
    PluginGroup.PARTITIONS: "players",
}
"""Plugin groups the catalogue must cover, with their ``explanation`` field."""


def _used(field: str) -> set[str]:
    used: set[str] = set()
    for scenario in SCENARIOS.values():
        model = MODELS[scenario.tier]
        for variant in [None, *scenario.variants]:
            value = getattr(scenario.config(model, variant).explanation, field)
            used.add(str(getattr(value, "value", value)))
    return used


@pytest.mark.parametrize("group", list(FIELDS), ids=lambda g: g.value)
def test_every_builtin_plugin_has_a_scenario(group: PluginGroup) -> None:
    """Each built-in of the group is the effective setting of some scenario."""
    missing = set(BUILTIN_PLUGINS[group]) - _used(FIELDS[group])
    assert not missing, f"no practical scenario uses {group.value}: {sorted(missing)}"


def test_every_scenario_has_a_test() -> None:
    """Each scenario id is quoted by a module that parametrizes ``scenario_cases``."""
    texts = (path.read_text(encoding="utf-8") for path in HERE.rglob("test_*.py"))
    sources = "\n".join(text for text in texts if "scenario_cases(" in text)
    missing = [i for i in SCENARIOS if f'"{i}"' not in sources]
    assert not missing, f"scenarios without a test: {missing}"
