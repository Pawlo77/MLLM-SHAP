"""Spectrogram-guided alignment of transcript tokens to speech (``audio`` extra).

@domain alignment
@see [[knowledge/architecture/alignment]]
"""

from .io import decode_audio, encode_wav
from .segments import AudioSegment
from .sgpa import CtcModel, SpectrogramGuidedAligner, Wav2Vec2Ctc

__all__ = [
    "AudioSegment",
    "CtcModel",
    "SpectrogramGuidedAligner",
    "Wav2Vec2Ctc",
    "decode_audio",
    "encode_wav",
]
