"""Spectrogram-guided alignment of transcript tokens to speech (SGPA).

A CTC model aligns the transcript's characters to frames; each token boundary then
moves to the quietest, steadiest point of the waveform near it (RMS energy and
spectral flux), so a segment cut does not split a sound.

@domain alignment
@see [[knowledge/architecture/alignment]]
@tested-by [[tests/alignment/test_sgpa.py]]
"""

import itertools
import logging
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..domain.conversation import AudioPart
from .ctc import forced_align, target_spans
from .io import decode_audio, encode_wav, require
from .segments import AudioSegment

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "facebook/wav2vec2-base-960h"
"""English character-level CTC model used when no model is given."""

MIN_DURATION_S: float = 0.05
"""Shortest segment the aligner emits, in seconds."""

_EDGE_WINDOW_S = 0.08  # search half-width around the first and last edge
_GAP_PAD_S = 0.04  # extra search width on each side of a gap between tokens
_FRAME, _HOP = 256, 64  # RMS and STFT window and hop, in samples
_SILENCE_RATIO = 0.5  # a cut needs RMS at most this share of the window's mean


class CtcModel(Protocol):
    """Character-level CTC acoustic model."""

    sample_rate: int
    """Sample rate the model expects."""

    blank_id: int
    """Id of the CTC blank label."""

    vocab: Mapping[str, int]
    """Label id of every character the model emits."""

    def log_probs(self, waveform: NDArray[np.float32]) -> NDArray[np.floating]:
        """Return ``[frames, labels]`` log-probabilities for a mono waveform."""


_TRAINING_ONLY_KEYS = {"wav2vec2.masked_spec_embed"}
"""Weights SpecAugment masks use in training; inference never reads them."""


class Wav2Vec2Ctc:
    """A Hugging Face ``Wav2Vec2ForCTC`` checkpoint (``transformers`` extra)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Load the processor and model; ``revision`` pins the Hub commit."""
        transformers = require("transformers")
        self._torch = require("torch")
        self._processor = transformers.Wav2Vec2Processor.from_pretrained(
            model_id, revision=revision
        )
        hf_logging = require("transformers.utils.logging")
        verbosity = hf_logging.get_verbosity()
        # Its load report always lists the training-only masked_spec_embed as missing.
        hf_logging.set_verbosity_error()
        try:
            model, info = transformers.Wav2Vec2ForCTC.from_pretrained(
                model_id, revision=revision, output_loading_info=True
            )
        finally:
            hf_logging.set_verbosity(verbosity)
        if unexpected := set(info["missing_keys"]) - _TRAINING_ONLY_KEYS:
            logger.warning(
                "%s: weights missing from checkpoint: %s", model_id, sorted(unexpected)
            )
        self._model = model.to(device).eval()
        self._device = device
        tokenizer = self._processor.tokenizer
        self.vocab: Mapping[str, int] = tokenizer.get_vocab()
        self.blank_id: int = tokenizer.pad_token_id or 0
        self.sample_rate: int = self._processor.feature_extractor.sampling_rate

    def log_probs(self, waveform: NDArray[np.float32]) -> NDArray[np.floating]:
        """Run the model once over ``waveform``."""
        torch = self._torch
        inputs = self._processor(
            waveform, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self._model(inputs.input_values.to(self._device)).logits[0]
            log_probs: NDArray[np.floating] = (
                torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            )
            return log_probs


class SpectrogramGuidedAligner:
    """Align transcript tokens to speech and refine their boundaries."""

    def __init__(
        self,
        model: CtcModel | None = None,
        energy_weight: float = 0.8,
        flux_weight: float = 0.2,
        separator: str = "|",
    ) -> None:
        """Configure the aligner.

        Args:
            model: The CTC model; default :class:`Wav2Vec2Ctc` with ``DEFAULT_MODEL``.
            energy_weight: Weight of normalised RMS energy in the boundary cost.
            flux_weight: Weight of normalised spectral flux in the boundary cost.
            separator: The model's word-separator character.

        Raises:
            ValueError: If a weight is negative or both are zero.
        """
        if min(energy_weight, flux_weight) < 0 or energy_weight + flux_weight <= 0:
            msg = "boundary weights must be non-negative with a positive sum"
            raise ValueError(msg)
        self.model = model if model is not None else Wav2Vec2Ctc()
        self._weights = (energy_weight, flux_weight)
        self._separator = separator

    def align(
        self,
        transcript: str | Sequence[str],
        audio: AudioPart,
        attach_audio: bool = False,
    ) -> list[AudioSegment]:
        """Return one segment per token of ``transcript`` the model can spell.

        A string is split on whitespace. Tokens without a character in the model's
        vocabulary (punctuation, for example) have no segment.

        Raises:
            ValueError: If no token has a character the model knows, or the audio
                is too short for the transcript.
        """
        tokens = transcript.split() if isinstance(transcript, str) else list(transcript)
        targets, owners = self._targets(tokens)
        waveform, rate = decode_audio(audio.data())
        model_rate = self.model.sample_rate
        model_wave = (
            waveform
            if rate == model_rate
            else require("librosa").resample(
                waveform, orig_sr=rate, target_sr=model_rate
            )
        )
        log_probs = self.model.log_probs(model_wave)
        spans = target_spans(
            forced_align(log_probs, targets, self.model.blank_id), log_probs, targets
        )
        seconds_per_frame = len(model_wave) / len(log_probs) / model_rate
        rough: dict[int, list[tuple[float, float, float]]] = {}
        for owner, span in zip(owners, spans, strict=True):
            if owner >= 0:
                rough.setdefault(owner, []).append((
                    span.start * seconds_per_frame,
                    span.end * seconds_per_frame,
                    span.confidence,
                ))
        kept = sorted(rough)
        edges = self._edges([rough[k] for k in kept], waveform, rate)
        source = waveform if attach_audio else None
        return [
            self._segment(tokens[k], k, rough[k], edges[i], edges[i + 1], source, rate)
            for i, k in enumerate(kept)
        ]

    def _targets(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        """Return the label ids to align and each label's token (-1: separator)."""
        vocab = self.model.vocab
        targets: list[int] = []
        owners: list[int] = []
        for index, token in enumerate(tokens):
            ids = [vocab[c] for c in _spellable(token, vocab, self._separator)]
            if not ids:
                logger.debug("token %r has no model characters; no segment", token)
                continue
            if targets and self._separator in vocab:
                targets.append(vocab[self._separator])
                owners.append(-1)
            targets.extend(ids)
            owners.extend([index] * len(ids))
        if not targets:
            msg = "the transcript has no characters the CTC model knows"
            raise ValueError(msg)
        return targets, owners

    def _edges(
        self,
        tokens: list[list[tuple[float, float, float]]],
        waveform: NDArray[np.float32],
        rate: int,
    ) -> list[tuple[float, bool]]:
        """Refined first start, shared token boundaries and last end."""
        edges = [self._refine(waveform, rate, tokens[0][0][0], _EDGE_WINDOW_S)]
        for left, right in itertools.pairwise(tokens):
            gap_start, gap_end = left[-1][1], right[0][0]
            time, refined = self._refine(
                waveform,
                rate,
                (gap_start + gap_end) / 2,
                (gap_end - gap_start) / 2 + _GAP_PAD_S,
            )
            edges.append((float(np.clip(time, gap_start, gap_end)), refined))
        end = min(tokens[-1][-1][1], len(waveform) / rate)
        edges.append(self._refine(waveform, rate, end, _EDGE_WINDOW_S))
        return edges

    def _refine(
        self, waveform: NDArray[np.float32], rate: int, time: float, half_width: float
    ) -> tuple[float, bool]:
        """Move ``time`` to the lowest-cost frame within ``half_width`` seconds."""
        first = max(0, int((time - half_width) * rate))
        region = waveform[first : int((time + half_width) * rate)]
        if len(region) < _FRAME:
            return time, False
        librosa = require("librosa")
        rms = librosa.feature.rms(y=region, frame_length=_FRAME, hop_length=_HOP)[0]
        stft = np.abs(librosa.stft(region, n_fft=_FRAME, hop_length=_HOP))
        flux = np.pad(np.sum(np.diff(stft, axis=1) ** 2, axis=0), (1, 0))[: len(rms)]
        energy, change = self._weights
        cost = energy * _normalised(rms) + change * _normalised(flux)
        best = int(np.argmin(cost))
        if rms[best] > _SILENCE_RATIO * float(np.mean(rms)):
            return time, False
        return (first + best * _HOP) / rate, True

    def _segment(
        self,
        token: str,
        index: int,
        chars: list[tuple[float, float, float]],
        start: tuple[float, bool],
        end: tuple[float, bool],
        waveform: NDArray[np.float32] | None,
        rate: int,
    ) -> AudioSegment:
        start_time, end_time = start[0], max(end[0], start[0] + MIN_DURATION_S)
        start_sample = round(start_time * rate)
        end_sample = max(round(end_time * rate), start_sample + 1)
        audio = b""
        if waveform is not None:
            audio = encode_wav(waveform[start_sample:end_sample], rate)
        return AudioSegment(
            token=token,
            token_index=index,
            start_time=start_time,
            end_time=end_time,
            ctc_start_time=chars[0][0],
            ctc_end_time=chars[-1][1],
            confidence=float(np.mean([confidence for _, _, confidence in chars])),
            boundary_refined=start[1] and end[1],
            sample_rate=rate,
            start_sample=start_sample,
            end_sample=end_sample,
            audio=audio,
        )


def _spellable(token: str, vocab: Mapping[str, int], separator: str) -> list[str]:
    """Characters of ``token`` the model emits, without diacritics, case-matched."""
    plain = "".join(
        c
        for c in unicodedata.normalize("NFD", token)
        if unicodedata.category(c) != "Mn"
    )
    chars = []
    for char in plain:
        match = char if char in vocab else char.upper()
        if match in vocab and match != separator:
            chars.append(match)
    return chars


def _normalised(values: NDArray[np.floating]) -> NDArray[np.floating]:
    scaled: NDArray[np.floating] = (values - values.min()) / (
        values.max() - values.min() + 1e-9
    )
    return scaled
