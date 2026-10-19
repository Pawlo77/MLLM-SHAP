"""Sphinx extension: the contract table of the built-in value functions.

``.. value-table::`` lists every built-in value function with the ``ValueInfo``
it declares: score range, direction, whether it reads the full-input answer,
required capabilities, extra model and cost per coalition.
"""

import math
import typing

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata

COSTS = {
    "text": "string work",
    "encoder": "one encoder pass",
    "forward": "one forward pass",
    "judge": "one judge call",
}
"""Meaning of each ``ValueInfo.cost`` tier."""


def _bound(value: float) -> str:
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return f"{value:g}"


class ValueTableDirective(SphinxDirective):
    """Render the declared ``ValueInfo`` of every built-in value function."""

    def run(self) -> list[nodes.Node]:
        """Build the table from the built-in plugins."""
        from mllm_shap.plugins import PluginGroup
        from mllm_shap.plugins.builtin import BUILTIN_PLUGINS

        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 16 12 8 8 18 26 12",
            "",
            "   * - ``value_fn``",
            "     - Range",
            "     - Reads :math:`y_N`",
            "     - Needs",
            "     - Extra model",
            "     - Cost per coalition",
            "     - Direction",
        ]
        for name, factory in sorted(BUILTIN_PLUGINS[PluginGroup.VALUE_FNS].items()):
            info = typing.cast(typing.Any, factory).info
            low, high = info.value_range
            lines.extend([
                f"   * - ``{name}``",
                f"     - [{_bound(low)}, {_bound(high)}]",
                f"     - {'yes' if info.reads_base else 'no'}",
                f"     - {', '.join(info.needs) or '-'}",
                f"     - {info.extra_model or '-'}",
                f"     - {COSTS[info.cost]}",
                f"     - {'higher is better' if info.higher_is_better else 'lower'}",
            ])
        container = nodes.container()
        self.state.nested_parse(StringList(lines, source=""), 0, container)
        return [container]


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the value-table directive."""
    app.add_directive("value-table", ValueTableDirective)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
