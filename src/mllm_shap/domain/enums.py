"""Conversation and explanation enumerations for the text MVP.

@domain domain
@see [[knowledge/architecture/domain]]
@tested-by [[tests/domain/test_enums.py]]
"""

from enum import StrEnum
from typing import Literal


class Role(StrEnum):
    """Participant role attached to a message or token."""

    USER = "user"
    """End-user or human-provided input."""

    ASSISTANT = "assistant"
    """Model-generated output."""

    SYSTEM = "system"
    """System-level instruction or steering text."""


RoleName = Literal["user", "assistant", "system"]
"""A :class:`Role` value as plugin options spell it."""


class EmbeddingMode(StrEnum):
    """Whether value functions use static or contextual embeddings."""

    STATIC = "static"
    """Embeddings computed from final generated tokens only."""

    CONTEXTUAL = "contextual"
    """Embeddings derived from model internal states."""
