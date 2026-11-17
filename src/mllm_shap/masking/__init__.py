"""Coalition masking codecs, views, and absence policies.

@domain masking
@see [[knowledge/architecture/masking]]
"""

from .builder import MaskBuilder, MaskedSnapshot
from .codec import MaskCodec, PackedMask
from .infill import InfillPolicy
from .partitions import (
    MessagePartitioner,
    MessagesOptions,
    SentencePartitioner,
    TokenPartitioner,
)
from .policies import DeletePolicy, NeutralPolicy, PadPolicy
from .space import MaskSpace

__all__ = [
    "DeletePolicy",
    "InfillPolicy",
    "MaskBuilder",
    "MaskCodec",
    "MaskSpace",
    "MaskedSnapshot",
    "MessagePartitioner",
    "MessagesOptions",
    "NeutralPolicy",
    "PackedMask",
    "PadPolicy",
    "SentencePartitioner",
    "TokenPartitioner",
]
