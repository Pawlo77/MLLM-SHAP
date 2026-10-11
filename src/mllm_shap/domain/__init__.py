"""Domain types for cooperative-game explainability.

@domain domain
@see [[knowledge/architecture/domain]]
"""

from .coalition import CoalitionMask
from .config import DedupConfig, GenerationConfig, KvCacheConfig
from .conversation import (
    AudioPart,
    ContentPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from .enums import EmbeddingMode, Role
from .estimands import Estimand, EstimandWire, estimand_to_wire
from .game import CooperativeGame
from .generation import GenerationRecord
from .players import PlayerSet, PlayerSpan

__all__ = [
    "AudioPart",
    "CoalitionMask",
    "ContentPart",
    "ConversationSnapshot",
    "CooperativeGame",
    "DedupConfig",
    "EmbeddingMode",
    "Estimand",
    "EstimandWire",
    "GenerationConfig",
    "GenerationRecord",
    "KvCacheConfig",
    "Message",
    "PlayerSet",
    "PlayerSpan",
    "Role",
    "TextPart",
    "Turn",
    "estimand_to_wire",
]
