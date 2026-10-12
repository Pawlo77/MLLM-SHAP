"""Tests for built-in plugin registration.

@see [[src/mllm_shap/plugins/builtin.py]]
"""

import pytest

from mllm_shap.backends.mock import MockBackend
from mllm_shap.errors import PluginConflictError
from mllm_shap.estimation.complementary import ComplementaryEstimator
from mllm_shap.estimation.exact import ExactEstimator
from mllm_shap.estimation.monte_carlo import MonteCarloEstimator
from mllm_shap.estimation.neyman import NeymanEstimator
from mllm_shap.estimation.normalizers import IdentityNormalizer, MinMaxNormalizer
from mllm_shap.masking.partitions import TokenPartitioner
from mllm_shap.masking.policies import DeletePolicy, NeutralPolicy, PadPolicy
from mllm_shap.pipeline.config import ExplainConfig
from mllm_shap.plugins import (
    PluginContext,
    PluginGroup,
    PluginRegistry,
    default_registry,
    register_builtin_plugins,
)
from mllm_shap.plugins.backends import instantiate_backend
from mllm_shap.value.tfidf import TfIdfCosineValue


def _context(registry: PluginRegistry) -> PluginContext:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"}
    })
    return PluginContext(config=config, registry=registry)


def test_builtin_plugins_resolve_from_config_keys() -> None:
    """ExplainConfig player and absence keys resolve through the registry."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.PARTITIONS)
    registry.load_entry_points(PluginGroup.ABSENCE_POLICIES)

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"players": "tokens", "absence_policy": "delete"},
    })

    partitioner = registry.resolve(
        PluginGroup.PARTITIONS, config.explanation.players, _context(registry)
    )
    policy = registry.resolve(
        PluginGroup.ABSENCE_POLICIES,
        config.explanation.absence_policy,
        _context(registry),
    )

    assert isinstance(partitioner, TokenPartitioner)
    assert isinstance(policy, DeletePolicy)


def test_builtin_registers_all_absence_policies() -> None:
    """Built-in registration exposes delete, pad, and neutral policies."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)

    assert registry.names(PluginGroup.ABSENCE_POLICIES) == ("delete", "neutral", "pad")
    assert isinstance(
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "pad", _context(registry)),
        PadPolicy,
    )
    assert isinstance(
        registry.resolve(PluginGroup.ABSENCE_POLICIES, "neutral", _context(registry)),
        NeutralPolicy,
    )


def test_builtin_registers_exact_estimator() -> None:
    """Built-in registration exposes the exact coalition enumerator."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.ESTIMATORS)

    estimator = registry.resolve(PluginGroup.ESTIMATORS, "exact", _context(registry))
    assert isinstance(estimator, ExactEstimator)
    assert estimator.name == "exact"


def test_builtin_registers_approximate_estimators() -> None:
    """Built-in registration exposes MC, complementary, and Neyman estimators."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.ESTIMATORS)

    assert registry.names(PluginGroup.ESTIMATORS) == (
        "complementary",
        "exact",
        "loo",
        "mc",
        "neyman_cc",
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "mc", _context(registry)),
        MonteCarloEstimator,
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "complementary", _context(registry)),
        ComplementaryEstimator,
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "neyman_cc", _context(registry)),
        NeymanEstimator,
    )


def test_builtin_registers_value_functions_and_normalizers() -> None:
    """Built-in registration exposes value functions and normalizers."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.VALUE_FNS)
    registry.load_entry_points(PluginGroup.NORMALIZERS)

    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {"value_fn": "tfidf_cosine", "normalizer": "min_max"},
    })

    value_fn = registry.resolve(
        PluginGroup.VALUE_FNS, config.explanation.value_fn, _context(registry)
    )
    normalizer = registry.resolve(
        PluginGroup.NORMALIZERS, config.explanation.normalizer, _context(registry)
    )

    assert isinstance(value_fn, TfIdfCosineValue)
    assert isinstance(normalizer, MinMaxNormalizer)
    assert registry.names(PluginGroup.VALUE_FNS) == (
        "embedding_cosine",
        "embedding_euclidean",
        "logprob",
        "reference_wer",
        "tfidf_cosine",
    )
    assert registry.names(PluginGroup.NORMALIZERS) == (
        "abs_sum",
        "identity",
        "min_max",
        "power_shift",
    )
    assert isinstance(
        registry.resolve(PluginGroup.NORMALIZERS, "identity", _context(registry)),
        IdentityNormalizer,
    )


def test_builtin_registers_mock_backend() -> None:
    """Built-in registration exposes the mock backend class."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    registry.load_entry_points(PluginGroup.BACKENDS)

    backend = instantiate_backend(_context(registry))
    assert isinstance(backend, MockBackend)
    assert backend.model_id == "stub"


def test_builtin_registration_matches_packaging_entry_points() -> None:
    """Every group's builtin names equal the names declared in pyproject.toml."""
    builtin = PluginRegistry()
    register_builtin_plugins(builtin)
    for group in PluginGroup:
        from_entry_points = PluginRegistry()
        from_entry_points.load_entry_points(group)
        assert builtin.names(group) == from_entry_points.names(group), group


def test_default_registry_resolves_every_group() -> None:
    """The default registry holds builtins for all eight plugin groups."""
    registry = default_registry()
    for group in PluginGroup:
        assert registry.names(group), group


def test_register_rejects_duplicate_names_unless_replaced() -> None:
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    with pytest.raises(PluginConflictError, match="exact"):
        registry.register(PluginGroup.ESTIMATORS, "exact", MonteCarloEstimator)
    registry.register(
        PluginGroup.ESTIMATORS, "exact", MonteCarloEstimator, replace=True
    )
    assert isinstance(
        registry.resolve(PluginGroup.ESTIMATORS, "exact", _context(registry)),
        MonteCarloEstimator,
    )


def test_load_entry_points_skips_builtin_duplicates() -> None:
    """Loading entry points after builtins keeps the builtin factories."""
    registry = PluginRegistry()
    register_builtin_plugins(registry)
    factory = registry.resolve_factory(PluginGroup.ESTIMATORS, "exact")
    registry.load_entry_points(PluginGroup.ESTIMATORS)
    assert registry.resolve_factory(PluginGroup.ESTIMATORS, "exact") is factory
