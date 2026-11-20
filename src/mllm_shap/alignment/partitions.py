"""Audio partitions: ``sgpa_segments`` and ``uniform_segments``.

One player per spoken word, or one per equal-duration piece (the matched control).

@domain alignment
@see [[knowledge/architecture/alignment]]
@tested-by [[tests/alignment/test_sgpa_players.py]]
"""

import re
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import Field

from ..domain.conversation import AudioPart, ConversationSnapshot, Message
from ..domain.enums import RoleName
from ..domain.players import PlayerSet, PlayerSpan
from ..protocols.plugin import PluginOptions
from .io import decode_audio
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

    text_roles: list[RoleName] = Field(default_factory=list)
    """Roles whose text-only messages also give one player per word, so typed and
    spoken words are explained in one run; none by default."""

    transcript: Literal["send", "align_only"] = "send"
    """``send`` gives the model each audio message's transcript with its audio;
    ``align_only`` uses the transcript to build the players and sends audio only,
    so values reflect what the model hears, not what it reads."""


class SgpaSegmentPartitioner:
    """Make each spellable word of every audio message's transcript a player.

    A message qualifies when it has an :class:`~mllm_shap.AudioPart` and text: the
    text is the transcript the audio is aligned to. Each player's span covers the
    word in the text and, through ``samples``, its stretch of audio, so absence
    policies mask both. Text-only messages give word players when their role is in
    ``text_roles`` and no players otherwise.
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
        hidden: list[int] = []
        message_index = 0
        for turn_index, turn in enumerate(snapshot.turns):
            for local_index, message in enumerate(turn.messages):
                prefix = f"{snapshot.snapshot_id}:{turn_index}:{local_index}"
                for token_index, span in self._players(message, message_index):
                    ids.append(f"{prefix}:{token_index}")
                    spans.append(span)
                if self._options.transcript == "align_only" and any(
                    isinstance(part, AudioPart) for part in message.content
                ):
                    hidden.append(message_index)
                message_index += 1
        if not ids:
            msg = "sgpa_segments needs a message with an AudioPart and its transcript"
            raise ValueError(msg)
        return PlayerSet(
            player_ids=tuple(ids), spans=tuple(spans), hidden_text=tuple(hidden)
        )

    def _players(
        self, message: Message, message_index: int
    ) -> list[tuple[int, PlayerSpan]]:
        audio = next((p for p in message.content if isinstance(p, AudioPart)), None)
        if audio is None:
            if message.role.value not in self._options.text_roles:
                return []
            return [
                (
                    index,
                    PlayerSpan(
                        message_index, w.start(), w.end(), w.group(), message.speaker
                    ),
                )
                for index, w in enumerate(_TOKEN_RE.finditer(message.text))
            ]
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


class UniformOptions(PluginOptions):
    """Options of the ``uniform_segments`` partition."""

    segments: int | None = Field(default=None, ge=1)
    """Players per audio message; the word count of its transcript when ``None``."""

    frame_ms: float = Field(default=20.0, gt=0.0)
    """Frame length in milliseconds; segment boundaries sit on whole frames."""


class UniformSegmentPartitioner:
    """Cut every audio message into equal-duration segments, one player each.

    The control of ``sgpa_segments``: with the transcript's word count as the
    default ``segments`` both make the same number of players, but these ignore
    where words are. Boundaries sit on whole ``frame_ms`` frames. The transcript is
    never sent (``PlayerSet.hidden_text``), so values reflect what the model hears;
    segment ``i`` takes word ``i`` as its text anchor.
    """

    options_model: ClassVar[type[PluginOptions]] = UniformOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext | None" = None) -> None:
        """Read the options from ``context`` when given."""
        self._options = UniformOptions()
        if context is not None:
            self._options = UniformOptions.model_validate(context.options)

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "uniform_segments"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Return the segments of every audio message as players.

        Raises:
            ValueError: If no message has audio, an audio message has neither a
                transcript nor ``segments``, or the audio has fewer frames than
                segments.
        """
        ids: list[str] = []
        spans: list[PlayerSpan] = []
        hidden: list[int] = []
        message_index = 0
        for turn_index, turn in enumerate(snapshot.turns):
            for local_index, message in enumerate(turn.messages):
                audio = next(
                    (p for p in message.content if isinstance(p, AudioPart)), None
                )
                if audio is not None:
                    prefix = f"{snapshot.snapshot_id}:{turn_index}:{local_index}"
                    for index, span in enumerate(
                        self._segments(message, audio, message_index)
                    ):
                        ids.append(f"{prefix}:{index}")
                        spans.append(span)
                    hidden.append(message_index)
                message_index += 1
        if not ids:
            msg = "uniform_segments needs a message with an AudioPart"
            raise ValueError(msg)
        return PlayerSet(
            player_ids=tuple(ids), spans=tuple(spans), hidden_text=tuple(hidden)
        )

    def _segments(
        self, message: Message, audio: AudioPart, message_index: int
    ) -> list[PlayerSpan]:
        words = list(_TOKEN_RE.finditer(message.text))
        count = self._options.segments or len(words)
        if count == 0:
            msg = "an audio message needs a transcript or the segments option"
            raise ValueError(msg)
        waveform, rate = decode_audio(audio.data())
        length = len(waveform)
        grid = max(1, round(rate * self._options.frame_ms / 1000.0))
        frames = length // grid
        if count > frames:
            msg = f"{count} segments need more than the {frames} frames of the audio"
            raise ValueError(msg)
        edges = [round(k * frames / count) * grid for k in range(count)] + [length]
        end_of_text = len(message.text)
        return [
            PlayerSpan(
                message_index=message_index,
                start=words[i].start() if i < len(words) else end_of_text,
                end=words[i].end() if i < len(words) else end_of_text,
                text=words[i].group() if i < len(words) else "",
                speaker=message.speaker,
                label=f"{1000 * edges[i] // rate}-{1000 * edges[i + 1] // rate} ms",
                samples=(edges[i], edges[i + 1]),
            )
            for i in range(count)
        ]
