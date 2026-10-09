"""Plugin registry exports.

@domain plugins
@see [[knowledge/architecture/plugins]]
"""

from .builtin import BUILTIN_PLUGINS, default_registry, register_builtin_plugins
from .context import PluginContext
from .groups import PluginGroup
from .options import check_plugin_options
from .registry import PluginFactory, PluginRegistry

__all__ = [
    "BUILTIN_PLUGINS",
    "PluginContext",
    "PluginFactory",
    "PluginGroup",
    "PluginRegistry",
    "check_plugin_options",
    "default_registry",
    "register_builtin_plugins",
]
