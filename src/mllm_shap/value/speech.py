"""Speech value functions: SpeechBERTScore on audio and the transcription adapter.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_speech.py]]
"""

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from pydantic import Field

from ..domain.generation import GenerationRecord
from ..errors import ConfigError
from ..protocols.backend import GenerationResult
from ..protocols.plugin import PluginOptions
from ..protocols.value import FittableValueFunction, ValueFunction, ValueInfo
from .encoders import Recognizer, SpeechEncoder, load_recognizer, load_speech_encoder

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


def _audio_path(result: GenerationResult, value_fn: str) -> str:
    path = result.audio_path if isinstance(result, GenerationRecord) else None
    if path is None:
        msg = (
            f"value_fn {value_fn!r} scores spoken answers, but the generation "
            "has no audio_path"
        )
        raise ConfigError(msg)
    return path


class SpeechBertScoreOptions(PluginOptions):
    """Options of the ``speech_bertscore`` value function."""

    encoder: str = "microsoft/wavlm-large"
    """Self-supervised speech model whose frames are matched."""

    layer: int = 14
    """Hidden layer used; scores depend on it, so it is recorded in the manifest."""

    revision: str = "main"
    """Hub revision of the model; pin a commit to reproduce runs."""


class SpeechBertScoreValue:
    r"""Score :math:`v(S)` as the SpeechBERTScore precision of the spoken answer.

    Every frame of the coalition's audio takes its best cosine match among the
    frames of the full-input answer's audio, and :math:`v(S)` is the mean
    (Saeki et al., 2024). The two recordings need not align or share a length.
    Records must carry ``audio_path``.
    """

    options_model: ClassVar[type[PluginOptions]] = SpeechBertScoreOptions
    """Options accepted from ``explanation.plugin_options``."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-1.0, 1.0),
        reads_base=True,
        needs=("audio",),
        extra_model="WavLM-style speech encoder (default wavlm-large)",
        cost="encoder",
    )

    def __init__(
        self,
        context: "PluginContext | None" = None,
        encoder: SpeechEncoder | None = None,
    ) -> None:
        """Read the options in ``context``; ``encoder`` injects a loaded encoder."""
        options = SpeechBertScoreOptions()
        if context is not None:
            options = SpeechBertScoreOptions.model_validate(context.options)
        self._options = options
        self._encoder = encoder
        self._frames: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "speech_bertscore"

    def manifest_info(self) -> dict[str, str]:
        """Return the encoder and layer for the run manifest."""
        return {
            "encoder": self._options.encoder,
            "layer": str(self._options.layer),
            "revision": self._options.revision,
        }

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the mean best-match cosine of ``candidate``'s frames."""
        reference = self._embed(_audio_path(base, self.name))
        heard = self._embed(_audio_path(candidate, self.name))
        if not len(reference) or not len(heard):
            return 0.0
        return float((heard @ reference.T).max(axis=1).mean())

    def _embed(self, path: str) -> np.ndarray:
        if path not in self._frames:
            if self._encoder is None:
                self._encoder = load_speech_encoder(
                    self._options.encoder, self._options.layer, self._options.revision
                )
            self._frames[path] = self._encoder.encode(path)
        return self._frames[path]


class TranscribedOptions(PluginOptions):
    """Options of the ``transcribed`` adapter."""

    value_fn: str
    """Text value function applied to the transcripts, such as ``bertscore``."""

    recognizer: str = "openai/whisper-base"
    """Hugging Face speech-recognition model that transcribes both answers."""

    inner_options: dict[str, object] = Field(default_factory=dict)
    """``plugin_options`` of the wrapped value function."""


class TranscribedValue:
    """Score spoken answers by transcribing them and applying a text value function.

    It answers content-only questions: the fixed recogniser removes how the
    answer sounds. The wrapped function sees records whose text is the transcript
    (``audio_path`` is dropped), and is fitted on the transcripts when it needs it.
    """

    options_model: ClassVar[type[PluginOptions]] = TranscribedOptions
    """Options accepted from ``explanation.plugin_options``."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-float("inf"), float("inf")),
        reads_base=True,
        needs=("audio",),
        extra_model="speech recognizer (default whisper-base) plus the wrapped one",
        cost="encoder",
    )

    def __init__(
        self,
        context: "PluginContext",
        recognizer: Recognizer | None = None,
    ) -> None:
        """Build the wrapped value function; ``recognizer`` injects a loaded one."""
        # Imported here: the plugin package imports this module.
        from ..plugins.groups import PluginGroup
        from ..plugins.options import validate_plugin_options

        options = TranscribedOptions.model_validate(context.options)
        factory = context.registry.resolve_factory(
            PluginGroup.VALUE_FNS, options.value_fn
        )
        inner_options = validate_plugin_options(
            factory, PluginGroup.VALUE_FNS, options.value_fn, options.inner_options
        )
        self._inner = cast(
            ValueFunction, factory(dataclasses.replace(context, options=inner_options))
        )
        self._options = options
        self._recognizer = recognizer
        self._transcripts: dict[str, str] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return "transcribed"

    def manifest_info(self) -> dict[str, str]:
        """Return the recognizer and wrapped value function for the manifest."""
        return {
            "recognizer": self._options.recognizer,
            "value_fn": self._options.value_fn,
        }

    def fit(self, corpus: Sequence[GenerationRecord]) -> None:
        """Transcribe every answer, then fit the wrapped value function on them."""
        if isinstance(self._inner, FittableValueFunction):
            self._inner.fit([self._transcribe(record) for record in corpus])

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the wrapped value function on the two transcripts."""
        return self._inner.score(self._transcribe(base), self._transcribe(candidate))

    def _transcribe(self, result: GenerationResult) -> GenerationRecord:
        from ..value.generation_records import generation_record_from_text

        path = _audio_path(result, self.name)
        if path not in self._transcripts:
            if self._recognizer is None:
                self._recognizer = load_recognizer(self._options.recognizer)
            self._transcripts[path] = self._recognizer.transcribe(path)
        return generation_record_from_text(self._transcripts[path])
