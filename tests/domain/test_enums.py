"""Tests for conversation domain enums.

@see [[src/nlp_shap/domain/enums.py]]
"""

import pytest

from nlp_shap.domain.enums import EmbeddingMode, Role


def test_role_wire_values_are_stable() -> None:
    """Role members serialize to stable lowercase strings."""
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.SYSTEM.value == "system"


def test_embedding_mode_values_are_stable() -> None:
    """Embedding mode members serialize to stable strings."""
    assert EmbeddingMode.STATIC.value == "static"
    assert EmbeddingMode.CONTEXTUAL.value == "contextual"


def test_role_rejects_unknown_value() -> None:
    """Unknown role wire values raise a lookup error."""
    with pytest.raises(ValueError, match="not a valid"):
        Role("moderator")
