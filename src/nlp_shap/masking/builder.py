"""View-based masked snapshots with structural sharing.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_builder.py]]
"""

from dataclasses import dataclass

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet
from ..protocols.masking import AbsencePolicy
from .codec import MaskCodec


@dataclass(frozen=True, slots=True)
class MaskedSnapshot:
    """Shared view over a conversation snapshot and coalition mask."""

    base: ConversationSnapshot
    """Underlying snapshot shared across coalition evaluations."""

    players: PlayerSet
    """Ordered players aligned with the coalition mask."""

    mask: CoalitionMask
    """Presence mask selecting which players remain in the coalition."""

    policy_name: str
    """Registered absence-policy identifier used to render the view."""

    def __post_init__(self) -> None:
        """Check that the mask matches the players."""
        self.mask.validate_against(self.players)

    @property
    def base_snapshot_id(self) -> str:
        """Return the shared base snapshot identifier."""
        return self.base.snapshot_id

    @property
    def mask_hash(self) -> int:
        """Return a stable hash for the coalition mask."""
        return MaskCodec.hash_mask(self.mask.present)


class MaskBuilder:
    """Build masked snapshot views without copying the base conversation."""

    def __init__(self, policy: AbsencePolicy) -> None:
        """Render coalitions with ``policy``."""
        self._policy = policy

    @property
    def policy_name(self) -> str:
        """Return the configured absence-policy identifier."""
        return self._policy.name

    def view(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> MaskedSnapshot:
        """Create a masked view that shares ``snapshot`` across coalitions."""
        mask.validate_against(players)
        return MaskedSnapshot(
            base=snapshot,
            players=players,
            mask=mask,
            policy_name=self._policy.name,
        )

    def render(self, masked: MaskedSnapshot) -> ConversationSnapshot:
        """Materialize a coalition-specific conversation snapshot."""
        if masked.policy_name != self._policy.name:
            msg = (
                f"masked view policy {masked.policy_name!r} does not match "
                f"builder policy {self._policy.name!r}"
            )
            raise ValueError(msg)
        return self._policy.apply(masked.base, masked.players, masked.mask)
