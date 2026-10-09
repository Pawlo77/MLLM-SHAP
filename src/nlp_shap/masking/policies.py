"""Absence policies for rendering masked conversation snapshots.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_policies.py]]
"""

from typing import TYPE_CHECKING, ClassVar

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot
from ..domain.players import PlayerSet, PlayerSpan
from ..protocols.plugin import PluginOptions
from .tokens import render_spans

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


def _require_spans(players: PlayerSet, mask: CoalitionMask) -> tuple[PlayerSpan, ...]:
    mask.validate_against(players)
    if not players.spans:
        msg = "player set has no spans; build it with a partitioner"
        raise ValueError(msg)
    return players.spans


class PadOptions(PluginOptions):
    """Options of the ``pad`` absence policy."""

    placeholder: str = "[MASK]"
    """Placeholder text inserted for absent players."""


class NeutralOptions(PluginOptions):
    """Options of the ``neutral`` absence policy."""

    fill_char: str = "_"
    """Single character repeated to match each absent player's length."""


class DeletePolicy:
    """Remove absent players from the rendered conversation."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Accept the plugin context; this plugin reads no settings from it."""
        _ = context

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "delete"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Drop absent players while preserving turn and message structure.

        The empty coalition keeps every message with empty player text (v(empty)).
        """
        spans = _require_spans(players, mask)
        return render_spans(
            snapshot,
            spans,
            [
                span.text if present else None
                for span, present in zip(spans, mask.present, strict=True)
            ],
        )


class PadPolicy:
    """Replace absent players with a fixed mask placeholder."""

    options_model: ClassVar[type[PluginOptions]] = PadOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        placeholder: str = "[MASK]",
    ) -> None:
        """Set the placeholder; option ``placeholder`` in ``context`` overrides it."""
        if context is not None:
            placeholder = context.option("placeholder", placeholder)
        self.placeholder = placeholder
        """Placeholder text inserted for absent players."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "pad"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Substitute absent players with :attr:`placeholder`."""
        spans = _require_spans(players, mask)
        return render_spans(
            snapshot,
            spans,
            [
                span.text if present else self.placeholder
                for span, present in zip(spans, mask.present, strict=True)
            ],
        )


class NeutralPolicy:
    """Replace absent players with neutral fillers that preserve their width."""

    options_model: ClassVar[type[PluginOptions]] = NeutralOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        fill_char: str = "_",
    ) -> None:
        """Set the filler; option ``fill_char`` in ``context`` overrides it."""
        if context is not None:
            fill_char = context.option("fill_char", fill_char)
        self.fill_char = fill_char
        """Single character repeated to match each absent player's length."""

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "neutral"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Substitute absent players with width-matched neutral fillers."""
        if len(self.fill_char) != 1:
            msg = "fill_char must be exactly one character"
            raise ValueError(msg)
        spans = _require_spans(players, mask)
        return render_spans(
            snapshot,
            spans,
            [
                span.text if present else self.fill_char * len(span.text)
                for span, present in zip(spans, mask.present, strict=True)
            ],
        )
