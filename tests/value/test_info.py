"""Every built-in value function declares what it measures and costs.

@see [[src/mllm_shap/protocols/value.py]]
"""

import math

import pytest

from mllm_shap.plugins import PluginGroup
from mllm_shap.plugins.builtin import BUILTIN_PLUGINS
from mllm_shap.protocols import ValueInfo

COSTS = {"text", "encoder", "forward", "judge"}
NEEDS = {"logprobs", "embeddings", "teacher_forcing", "judge", "audio"}
NAMES = sorted(BUILTIN_PLUGINS[PluginGroup.VALUE_FNS])


@pytest.mark.parametrize("name", NAMES)
def test_builtin_value_function_declares_its_contract(name: str) -> None:
    factory = BUILTIN_PLUGINS[PluginGroup.VALUE_FNS][name]
    info = getattr(factory, "info", None)

    assert isinstance(info, ValueInfo), f"{name} declares no ValueInfo"
    low, high = info.value_range
    assert low < high or (low == high == 0.0) or math.isinf(low) or math.isinf(high)
    assert info.cost in COSTS
    assert set(info.needs) <= NEEDS
    if "teacher_forcing" in info.needs:
        assert info.cost == "forward"
    if info.cost in {"encoder", "judge"}:
        assert info.extra_model or "audio" in info.needs
