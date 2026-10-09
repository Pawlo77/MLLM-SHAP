"""Context handed to every plugin factory.

@domain plugins
@see [[knowledge/architecture/plugins]]
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..errors import ConfigError
from .registry import PluginRegistry

if TYPE_CHECKING:
    from ..pipeline.config import ExplainConfig


@dataclass(frozen=True, slots=True)
class PluginContext:
    """Inputs a plugin factory receives when the registry builds a plugin.

    Plugin classes take the context as their first constructor argument
    (``Plugin(context)``) and read the settings they need from it.
    """

    config: "ExplainConfig"
    """Validated explain configuration of the run."""

    options: Mapping[str, object] = field(default_factory=dict)
    """Validated options for this plugin from ``explanation.plugin_options``.

    Looked up under ``<group>/<name>`` first, then under the bare plugin name.
    """

    registry: PluginRegistry = field(default_factory=PluginRegistry)
    """Registry that built the plugin, for plugins that resolve other plugins."""

    def option[T](self, key: str, default: T) -> T:
        """Return ``options[key]``, or ``default`` when the key is absent.

        Integers are accepted where the default is a float.

        Raises:
            ConfigError: If the option value has a different type than ``default``.
        """
        value = self.options.get(key, default)
        if (
            isinstance(default, float)
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            value = float(value)
        if not isinstance(value, type(default)):
            msg = (
                f"plugin option {key!r} must be {type(default).__name__}, "
                f"got {type(value).__name__}"
            )
            raise ConfigError(msg)
        return value
