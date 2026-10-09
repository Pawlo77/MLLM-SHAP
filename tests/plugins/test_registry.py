"""Tests for plugin registry entry-point loading.

@see [[src/nlp_shap/plugins/registry.py]]
"""

import pytest

from nlp_shap.backends.mock import MockBackend
from nlp_shap.domain.generation import GenerationRecord
from nlp_shap.errors import ConfigError, PluginLoadError, PluginNotFoundError
from nlp_shap.estimation.estimands.banzhaf import BanzhafAggregator
from nlp_shap.estimation.estimands.shapley import ShapleyAggregator
from nlp_shap.estimation.neyman import NeymanEstimator
from nlp_shap.estimation.normalizers import IdentityNormalizer, PowerShiftNormalizer
from nlp_shap.masking.policies import PadPolicy
from nlp_shap.pipeline.config import BackendConfig, ExplainConfig
from nlp_shap.plugins import (
    PluginContext,
    PluginGroup,
    PluginRegistry,
    check_plugin_options,
    default_registry,
    register_builtin_plugins,
)
from nlp_shap.plugins.backends import instantiate_backend
from nlp_shap.value.embedding import CosineEmbeddingValue
from tests.backends._context import backend_context


def _context(registry: PluginRegistry) -> PluginContext:
    config = ExplainConfig(backend=BackendConfig(kind="mock", model_id="stub"))
    return PluginContext(config=config, registry=registry)


def test_registry_loads_estimand_entry_points() -> None:
    """Packaging entry points resolve to concrete estimand aggregators."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.ESTIMANDS)

    shapley = registry.resolve(PluginGroup.ESTIMANDS, "shapley", _context(registry))
    banzhaf = registry.resolve(PluginGroup.ESTIMANDS, "banzhaf", _context(registry))

    assert isinstance(shapley, ShapleyAggregator)
    assert isinstance(banzhaf, BanzhafAggregator)
    assert registry.names(PluginGroup.ESTIMANDS) == ("banzhaf", "shapley")


def test_registry_loads_value_fn_and_normalizer_entry_points() -> None:
    """Packaging entry points resolve to value functions and normalizers."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.VALUE_FNS)
    registry.load_entry_points(PluginGroup.NORMALIZERS)

    value_fn = registry.resolve(
        PluginGroup.VALUE_FNS, "tfidf_cosine", _context(registry)
    )
    normalizer = registry.resolve(
        PluginGroup.NORMALIZERS, "identity", _context(registry)
    )

    assert value_fn.name == "tfidf_cosine"
    assert normalizer.name == "identity"
    assert "logprob" in registry.names(PluginGroup.VALUE_FNS)
    assert "min_max" in registry.names(PluginGroup.NORMALIZERS)


def test_registry_loads_backend_entry_points() -> None:
    """Packaging entry points resolve to backend constructors."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.BACKENDS)

    backend_type = registry.resolve_type(PluginGroup.BACKENDS, "mock")
    backend = backend_type(backend_context(BackendConfig(kind="mock", model_id="stub")))
    assert isinstance(backend, MockBackend)
    assert backend.model_id == "stub"
    assert "lmstudio" in registry.names(PluginGroup.BACKENDS)
    assert "transformers" in registry.names(PluginGroup.BACKENDS)
    assert "api" in registry.names(PluginGroup.BACKENDS)


def test_registry_loads_renderer_entry_points() -> None:
    """Packaging entry points resolve renderer plugins."""
    registry = PluginRegistry()
    registry.load_entry_points(PluginGroup.RENDERERS)
    assert registry.names(PluginGroup.RENDERERS) == ("token_bar", "token_text")


def test_registry_register_and_resolve_round_trip() -> None:
    """Manual registration resolves to the registered factory."""
    registry = PluginRegistry()
    sentinel = object()
    registry.register(PluginGroup.VALUE_FNS, "stub", lambda context: sentinel)
    assert (
        registry.resolve(PluginGroup.VALUE_FNS, "stub", _context(registry)) is sentinel
    )


def test_registry_resolve_unknown_plugin_raises() -> None:
    """Unknown plugin names raise a lookup error."""
    registry = PluginRegistry()
    with pytest.raises(LookupError, match="missing"):
        registry.resolve(PluginGroup.BACKENDS, "missing", _context(registry))


def _configured_context(
    registry: PluginRegistry, **explanation: object
) -> PluginContext:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "from-config"},
        "explanation": explanation,
    })
    return PluginContext(config=config, registry=registry)


def test_resolve_passes_plugin_options_by_name() -> None:
    """A factory receives the options stored under its own plugin name."""
    registry = PluginRegistry()
    received: list[PluginContext] = []

    def factory(context: PluginContext) -> object:
        received.append(context)
        return object()

    registry.register(PluginGroup.VALUE_FNS, "custom", factory)
    context = _configured_context(
        registry,
        plugin_options={"custom": {"alpha": 2}, "other": {"beta": 3}},
    )
    registry.resolve(PluginGroup.VALUE_FNS, "custom", context)

    assert dict(received[0].options) == {"alpha": 2}
    assert received[0].registry is registry
    assert received[0].config is context.config


def test_neyman_estimator_reads_neyman_config() -> None:
    """``explanation.neyman`` reaches the estimator built by the registry."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(
        registry,
        neyman={"initial_num_samples": 4, "use_standard_method": True},
    )
    estimator = registry.resolve(PluginGroup.ESTIMATORS, "neyman_cc", context)

    assert isinstance(estimator, NeymanEstimator)
    assert estimator.initial_num_samples == 4
    assert estimator.use_standard_method is True


def test_embedding_value_reads_embedding_mode() -> None:
    """``explanation.embedding_mode`` selects contextual embeddings."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(registry, embedding_mode="contextual")
    value_fn = registry.resolve(PluginGroup.VALUE_FNS, "embedding_cosine", context)
    base = GenerationRecord(
        text="a", embedding=(1.0, 0.0), contextual_embedding=(0.0, 1.0)
    )
    candidate = GenerationRecord(
        text="b", embedding=(1.0, 0.0), contextual_embedding=(1.0, 0.0)
    )

    assert isinstance(value_fn, CosineEmbeddingValue)
    assert value_fn.score(base, candidate) == pytest.approx(0.0)


def test_backend_reads_backend_config_from_context() -> None:
    """Backends take their settings from ``context.config.backend``."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    backend = instantiate_backend(_configured_context(registry))

    assert isinstance(backend, MockBackend)
    assert backend.model_id == "from-config"


def test_pad_policy_reads_placeholder_option() -> None:
    """Plugin options override builtin constructor defaults."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(
        registry, plugin_options={"pad": {"placeholder": "<unk>"}}
    )
    policy = registry.resolve(PluginGroup.ABSENCE_POLICIES, "pad", context)

    assert isinstance(policy, PadPolicy)
    assert policy.placeholder == "<unk>"


def test_plugin_option_with_wrong_type_raises_config_error() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(registry, plugin_options={"pad": {"placeholder": 1}})

    with pytest.raises(ConfigError, match="placeholder"):
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "pad", context)


def test_unknown_plugin_raises_plugin_not_found() -> None:
    registry = PluginRegistry()
    with pytest.raises(PluginNotFoundError, match="missing"):
        registry.resolve(PluginGroup.BACKENDS, "missing", _context(registry))


def test_load_entry_points_wraps_load_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken third-party entry point surfaces its name in PluginLoadError."""

    class BrokenEntryPoint:
        name = "broken"
        value = "missing_package:Plugin"

        def load(self) -> object:
            msg = "No module named 'missing_package'"
            raise ModuleNotFoundError(msg)

    monkeypatch.setattr(
        "nlp_shap.plugins.registry.entry_points",
        lambda group: [BrokenEntryPoint()],
    )
    registry = PluginRegistry()

    with pytest.raises(PluginLoadError, match="broken") as excinfo:
        registry.load_entry_points(PluginGroup.VALUE_FNS)
    assert excinfo.value.entry_point_name == "broken"


def test_unknown_plugin_option_key_raises_config_error() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(
        registry, plugin_options={"pad": {"placeholder": "<unk>", "colour": "red"}}
    )

    with pytest.raises(ConfigError, match=r"absence_policies/pad.*'colour'"):
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "pad", context)


def test_builtin_without_options_rejects_any_option() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(registry, plugin_options={"delete": {"x": 1}})

    with pytest.raises(ConfigError, match=r"absence_policies/delete.*'x'"):
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "delete", context)


def test_plugin_options_accept_group_qualified_keys() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(
        registry,
        normalizer="power_shift",
        plugin_options={"normalizers/power_shift": {"power": 2}},
    )
    normalizer = registry.resolve(PluginGroup.NORMALIZERS, "power_shift", context)

    assert isinstance(normalizer, PowerShiftNormalizer)
    assert normalizer.power == 2.0


def test_plugin_options_for_one_plugin_under_two_keys_raise() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    context = _configured_context(
        registry,
        absence_policy="pad",
        plugin_options={
            "pad": {"placeholder": "a"},
            "absence_policies/pad": {"placeholder": "b"},
        },
    )

    with pytest.raises(ConfigError, match="both"):
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "pad", context)


def test_check_plugin_options_rejects_names_outside_the_run() -> None:
    registry = default_registry()
    context = _configured_context(
        registry, absence_policy="delete", plugin_options={"pad": {"placeholder": "x"}}
    )

    with pytest.raises(ConfigError, match=r"'pad'.*no plugin"):
        check_plugin_options(context.config, registry)


def test_check_plugin_options_rejects_unknown_group() -> None:
    registry = default_registry()
    context = _configured_context(
        registry, plugin_options={"policies/pad": {"placeholder": "x"}}
    )

    with pytest.raises(ConfigError, match="policies/pad"):
        check_plugin_options(context.config, registry)


def test_check_plugin_options_rejects_ambiguous_bare_names() -> None:
    registry = default_registry()
    registry.register(PluginGroup.NORMALIZERS, "pad", IdentityNormalizer)
    context = _configured_context(
        registry,
        absence_policy="pad",
        normalizer="pad",
        plugin_options={"pad": {"placeholder": "x"}},
    )

    with pytest.raises(ConfigError, match="ambiguous"):
        check_plugin_options(context.config, registry)


def test_check_plugin_options_validates_keys_before_the_run() -> None:
    registry = default_registry()
    context = _configured_context(
        registry,
        absence_policy="pad",
        plugin_options={"pad": {"placeholder_text": "x"}},
    )

    with pytest.raises(ConfigError, match=r"absence_policies/pad.*'placeholder_text'"):
        check_plugin_options(context.config, registry)


def test_check_plugin_options_accepts_renderer_options() -> None:
    registry = default_registry()
    context = _configured_context(registry, plugin_options={"renderers/token_bar": {}})
    check_plugin_options(context.config, registry)
