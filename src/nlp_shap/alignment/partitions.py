"""The ``sgpa_segments`` partition: one player per spoken word of an audio message.

@domain alignment
@see [[knowledge/architecture/alignment]]
@tested-by [[tests/alignment/test_sgpa_players.py]]
"""

import re
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

from ..domain.conversation import AudioPart, ConversationSnapshot, Message
from ..domain.players import PlayerSet, PlayerSpan
from ..protocols.plugin import PluginOptions
from .sgpa import DEFAULT_MODEL, SpectrogramGuidedAligner, Wav2Vec2Ctc

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

_TOKEN_RE = re.compile(r"\S+")


class SgpaOptions(PluginOptions):
    """Options of the ``sgpa_segments`` partition."""

    model_id: str = DEFAULT_MODEL
    """Hugging Face id of the Wav2Vec2 CTC model."""

    revision: str | None = None
    """Hub commit to load; pin it for reproducible players."""

    device: str = "cpu"
    """Torch device the model runs on."""

    energy_weight: float = Field(default=0.8, ge=0.0)
    """Weight of RMS energy in the boundary cost."""

    flux_weight: float = Field(default=0.2, ge=0.0)
    """Weight of spectral flux in the boundary cost."""


class SgpaSegmentPartitioner:
    """Make each spellable word of every audio message's transcript a player.

    A message qualifies when it has an :class:`~nlp_shap.AudioPart` and text: the
    text is the transcript the audio is aligned to. Each player's span covers the
    word in the text and, through ``samples``, its stretch of audio, so absence
    policies mask both. Other messages contribute no players.
    """

    options_model: ClassVar[type[PluginOptions]] = SgpaOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        aligner: SpectrogramGuidedAligner | None = None,
    ) -> None:
        """Use ``aligner``, or build one from the options on first use."""
        options = SgpaOptions()
        if context is not None:
            options = SgpaOptions.model_validate(context.options)
        self._options = options
        self._aligner = aligner

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "sgpa_segments"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Align every audio message and return its words as players.

        Raises:
            ValueError: If no message has audio, or an audio message has no
                transcript text.
        """
        ids: list[str] = []
        spans: list[PlayerSpan] = []
        message_index = 0
        for turn_index, turn in enumerate(snapshot.turns):
            for local_index, message in enumerate(turn.messages):
                prefix = f"{snapshot.snapshot_id}:{turn_index}:{local_index}"
                for token_index, span in self._players(message, message_index):
                    ids.append(f"{prefix}:{token_index}")
                    spans.append(span)
                message_index += 1
        if not ids:
            msg = "sgpa_segments needs a message with an AudioPart and its transcript"
            raise ValueError(msg)
        return PlayerSet(player_ids=tuple(ids), spans=tuple(spans))

    def _players(
        self, message: Message, message_index: int
    ) -> list[tuple[int, PlayerSpan]]:
        audio = next((p for p in message.content if isinstance(p, AudioPart)), None)
        if audio is None:
            return []
        words = list(_TOKEN_RE.finditer(message.text))
        if not words:
            msg = "an audio message needs its transcript as text for sgpa_segments"
            raise ValueError(msg)
        segments = self._get_aligner().align([w.group() for w in words], audio)
        return [
            (
                segment.token_index,
                PlayerSpan(
                    message_index=message_index,
                    start=words[segment.token_index].start(),
                    end=words[segment.token_index].end(),
                    text=segment.token,
                    speaker=message.speaker,
                    samples=(segment.start_sample, segment.end_sample),
                ),
            )
            for segment in segments
        ]

    def _get_aligner(self) -> SpectrogramGuidedAligner:
        if self._aligner is None:
            options = self._options
            model = Wav2Vec2Ctc(options.model_id, options.revision, options.device)
            self._aligner = SpectrogramGuidedAligner(
                model, options.energy_weight, options.flux_weight
            )
        return self._aligner
