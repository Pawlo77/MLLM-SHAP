"""Lookup and validation of ``explanation.plugin_options``.

@domain plugins
@see [[knowledge/architecture/plugins]]
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..errors import ConfigError
from ..protocols.plugin import DeclaresPluginOptions
from .groups import PluginGroup

if TYPE_CHECKING:
    from ..pipeline.config import ExplainConfig
    from .registry import PluginFactory, PluginRegistry


def group_key(group: PluginGroup | str) -> str:
    """Return the short group name used in qualified option keys (``value_fns``)."""
    return str(group).removeprefix("nlp_shap.")


def select_plugin_options(
    plugin_options: Mapping[str, Mapping[str, object]],
    group: PluginGroup | str,
    name: str,
) -> Mapping[str, object]:
    """Return the options stored under ``<group>/<name>`` or the bare ``<name>``.

    Raises:
        ConfigError: If both keys are present.
    """
    qualified = f"{group_key(group)}/{name}"
    if qualified in plugin_options and name in plugin_options:
        msg = (
            f"plugin_options has both {qualified!r} and {name!r} for the same "
            "plugin; keep one"
        )
        raise ConfigError(msg)
    return plugin_options.get(qualified, plugin_options.get(name, {}))


def validate_plugin_options(
    factory: "PluginFactory",
    group: PluginGroup | str,
    name: str,
    options: Mapping[str, object],
) -> Mapping[str, object]:
    """Check ``options`` against the factory's declared options model.

    Factories that declare no :class:`~nlp_shap.protocols.plugin.PluginOptions`
    model get ``options`` back unchanged.

    Raises:
        ConfigError: Naming the group, plugin and key of an unknown option or of
            a value with the wrong type.
    """
    if not isinstance(factory, DeclaresPluginOptions):
        return options
    try:
        validated = factory.options_model.model_validate(dict(options))
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in detail['loc'])!r}: {detail['msg']}"
            for detail in error.errors()
        )
        msg = f"invalid plugin_options for {group_key(group)}/{name}: {problems}"
        raise ConfigError(msg) from error
    return validated.model_dump(exclude_unset=True)


def run_plugins(config: "ExplainConfig") -> dict[str, str]:
    """Return the plugin name the run selects in each group, keyed by short group."""
    explanation = config.explanation
    return {
        group_key(PluginGroup.ESTIMATORS): explanation.estimator,
        group_key(PluginGroup.ESTIMANDS): explanation.estimand.value,
        group_key(PluginGroup.VALUE_FNS): explanation.value_fn,
        group_key(PluginGroup.NORMALIZERS): explanation.normalizer,
        group_key(PluginGroup.BACKENDS): config.backend.kind,
        group_key(PluginGroup.PARTITIONS): explanation.players,
        group_key(PluginGroup.ABSENCE_POLICIES): explanation.absence_policy,
    }


def check_plugin_options(config: "ExplainConfig", registry: "PluginRegistry") -> None:
    """Check every ``explanation.plugin_options`` entry before a run starts.

    A key is ``<group>/<name>`` (for example ``absence_policies/pad``) or a bare
    plugin name. It must name a plugin the run uses, or a registered renderer;
    a bare name must match one group only. Options of plugins that declare an
    options model are validated against it.

    Raises:
        ConfigError: If a key names no plugin of the run, an unknown group or
            more than one plugin, or holds an unknown option.
    """
    selected = run_plugins(config)
    renderers = set(registry.names(PluginGroup.RENDERERS))
    renderer_group = group_key(PluginGroup.RENDERERS)
    for key, options in config.explanation.plugin_options.items():
        if "/" in key:
            group, _, name = key.partition("/")
            if group == renderer_group:
                matched = name in renderers
            elif group in selected:
                matched = selected[group] == name
            else:
                msg = (
                    f"plugin_options key {key!r} names unknown group {group!r}; "
                    f"use one of {sorted([*selected, renderer_group])}"
                )
                raise ConfigError(msg)
            if not matched:
                msg = f"plugin_options key {key!r} matches no plugin of this run"
                raise ConfigError(msg)
            groups = [group]
        else:
            name = key
            groups = [group for group, chosen in selected.items() if chosen == name]
            if name in renderers:
                groups.append(renderer_group)
            if not groups:
                msg = (
                    f"plugin_options key {key!r} matches no plugin of this run; "
                    f"selected plugins: {selected}"
                )
                raise ConfigError(msg)
            if len(groups) > 1:
                msg = (
                    f"plugin_options key {key!r} is ambiguous (groups {groups}); "
                    f"use '<group>/{name}'"
                )
                raise ConfigError(msg)
        group = groups[0]
        full_group = f"nlp_shap.{group}"
        select_plugin_options(config.explanation.plugin_options, full_group, name)
        validate_plugin_options(
            registry.resolve_factory(full_group, name), full_group, name, options
        )
