"""Plugin registry with entry-point discovery.

@domain plugins
@see [[knowledge/architecture/plugins]]
@tested-by [[tests/plugins/test_registry.py]]
"""

import dataclasses
import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, cast

from ..errors import PluginConflictError, PluginLoadError, PluginNotFoundError
from .groups import PluginGroup
from .options import select_plugin_options, validate_plugin_options

if TYPE_CHECKING:
    from .context import PluginContext

logger = logging.getLogger(__name__)

PluginFactory = Callable[["PluginContext"], object]
"""Callable building a plugin from a
:class:`~mllm_shap.plugins.context.PluginContext`."""


class PluginRegistry:
    """Register and resolve pipeline plugins by group and name."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._factories: dict[str, dict[str, PluginFactory]] = {}

    def register(
        self,
        group: PluginGroup | str,
        name: str,
        factory: PluginFactory,
        replace: bool = False,
    ) -> None:
        """Register a plugin factory under ``group`` and ``name``.

        Raises:
            PluginConflictError: If ``name`` is already registered in ``group``
                and ``replace`` is false.
        """
        group_key = str(group)
        factories = self._factories.setdefault(group_key, {})
        if name in factories and not replace:
            msg = (
                f"plugin {name!r} is already registered in group {group_key!r}; "
                "pass replace=True to override it"
            )
            raise PluginConflictError(msg)
        factories[name] = factory

    def resolve(
        self,
        group: PluginGroup | str,
        name: str,
        context: "PluginContext",
    ) -> object:
        """Build the plugin registered under ``group`` and ``name``.

        The factory receives ``context`` with ``options`` set to
        ``explanation.plugin_options["<group>/<name>"]`` or, failing that,
        ``explanation.plugin_options["<name>"]`` (empty when both are absent),
        validated against the plugin's options model when it declares one.

        Raises:
            PluginNotFoundError: If nothing is registered under that name.
            ConfigError: If the options hold an unknown key or a wrong type, or
                the plugin has options under both keys.
        """
        factory = self.resolve_factory(group, name)
        options = select_plugin_options(
            context.config.explanation.plugin_options, group, name
        )
        options = validate_plugin_options(factory, group, name, options)
        return factory(dataclasses.replace(context, options=options))

    def resolve_factory(self, group: PluginGroup | str, name: str) -> PluginFactory:
        """Return the factory registered under ``group`` and ``name``.

        Raises:
            PluginNotFoundError: If nothing is registered under that name.
        """
        group_key = str(group)
        try:
            return self._factories[group_key][name]
        except KeyError as error:
            msg = f"unknown plugin {name!r} in group {group_key!r}"
            raise PluginNotFoundError(msg) from error

    def resolve_type(self, group: PluginGroup | str, name: str) -> type:
        """Return a plugin class registered under ``group`` and ``name``."""
        factory = self.resolve_factory(group, name)
        if not isinstance(factory, type):
            msg = f"plugin {name!r} in group {group!r} is not a class constructor"
            raise TypeError(msg)
        return factory

    def names(self, group: PluginGroup | str) -> tuple[str, ...]:
        """Return sorted plugin names registered for ``group``."""
        group_key = str(group)
        return tuple(sorted(self._factories.get(group_key, {})))

    def load_entry_points(self, group: PluginGroup | str) -> None:
        """Discover and register plugins from a packaging entry-point group.

        Names that are already registered (for example the built-in plugins
        registered by :func:`~mllm_shap.plugins.register_builtin_plugins`) are
        skipped without importing their entry point.

        Raises:
            PluginLoadError: If an entry point cannot be imported.
        """
        group_key = str(group)
        registered = self._factories.get(group_key, {})
        for entry_point in entry_points(group=group_key):
            if entry_point.name in registered:
                logger.debug(
                    "skipping entry point %r in %r: name already registered",
                    entry_point.name,
                    group_key,
                )
                continue
            try:
                target = entry_point.load()
            except Exception as error:
                msg = (
                    f"failed to load plugin entry point {entry_point.name!r} "
                    f"({entry_point.value}) in group {group_key!r}: {error}"
                )
                raise PluginLoadError(msg, entry_point.name) from error
            self.register(group_key, entry_point.name, cast(PluginFactory, target))
