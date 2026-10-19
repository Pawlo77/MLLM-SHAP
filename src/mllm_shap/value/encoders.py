"""Lazy loaders of the sentence and token encoders behind semantic value functions.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_semantic.py]]
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..errors import BackendUnavailableError

AUDIO_EXTRA = "pip install 'mllm-shap[audio-semantic]'"
"""Install hint shown when a speech encoder library is missing."""

SEMANTIC_EXTRA = "pip install 'mllm-shap[semantic]'"
"""Install hint shown when an encoder library is missing."""


class SentenceEncoder(Protocol):
    """Map texts to one vector each."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        """Return an ``(len(texts), dim)`` array."""


class TokenEncoder(Protocol):
    """Map texts to one matrix of unit-length token vectors each."""

    def encode(self, texts: Sequence[str]) -> list[NDArray[np.floating]]:
        """Return an ``(n_tokens, dim)`` array per text, special tokens removed."""


class _SentenceTransformer:
    def __init__(self, model: Any) -> None:
        self._model = model

    def encode(self, texts: Sequence[str]) -> NDArray[np.floating]:
        return np.asarray(self._model.encode(list(texts), show_progress_bar=False))


@lru_cache(maxsize=4)
def load_sentence_encoder(name: str) -> SentenceEncoder:
    """Load a sentence-transformers model once per process.

    Raises:
        BackendUnavailableError: If ``sentence-transformers`` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        msg = f"encoder {name!r} needs sentence-transformers; {SEMANTIC_EXTRA}"
        raise BackendUnavailableError(msg) from exc
    return _SentenceTransformer(SentenceTransformer(name))


class _HiddenStateEncoder:
    """Token vectors from one hidden layer of a Hugging Face encoder."""

    def __init__(self, tokenizer: Any, model: Any, layer: int) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._layer = layer

    def encode(
        self, texts: Sequence[str]
    ) -> list[NDArray[np.floating]]:  # pragma: no cover
        import torch

        batch = self._tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
        )
        special = batch.pop("special_tokens_mask").bool()
        with torch.no_grad():
            hidden = self._model(**batch, output_hidden_states=True).hidden_states[
                self._layer
            ]
        hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
        keep = batch["attention_mask"].bool() & ~special
        return [hidden[i][keep[i]].cpu().numpy() for i in range(len(texts))]


@lru_cache(maxsize=4)
def load_token_encoder(name: str, layer: int, revision: str = "main") -> TokenEncoder:
    """Load a Hugging Face encoder and expose layer ``layer`` as token vectors.

    Raises:
        BackendUnavailableError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        msg = f"encoder {name!r} needs transformers; {SEMANTIC_EXTRA}"
        raise BackendUnavailableError(msg) from exc
    model = AutoModel.from_pretrained(name, revision=revision).eval()
    return _HiddenStateEncoder(
        AutoTokenizer.from_pretrained(name, revision=revision), model, layer
    )


class WordEncoder(Protocol):
    """Map word lists to one contextual vector per word."""

    def encode(self, sentences: Sequence[Sequence[str]]) -> list[NDArray[np.floating]]:
        """Return an ``(n_words, dim)`` array per sentence."""


class _WordPoolEncoder:
    """Contextual word vectors: sub-word states of the last layer, mean-pooled."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self._tokenizer = tokenizer
        self._model = model

    def encode(
        self, sentences: Sequence[Sequence[str]]
    ) -> list[NDArray[np.floating]]:  # pragma: no cover
        import torch

        batch = self._tokenizer(
            [list(words) for words in sentences],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            hidden = self._model(**{
                k: v for k, v in batch.items() if k != "token_type_ids"
            }).last_hidden_state.float()
        out = []
        for i, words in enumerate(sentences):
            ids = batch.word_ids(i)
            vectors = np.zeros((len(words), hidden.shape[-1]), dtype=np.float32)
            counts = np.zeros(len(words))
            for position, word in enumerate(ids):
                if word is not None:
                    vectors[word] += hidden[i, position].cpu().numpy()
                    counts[word] += 1
            out.append((vectors / np.maximum(counts, 1)[:, None]).astype(np.float32))
        return out


@lru_cache(maxsize=2)
def load_word_encoder(name: str, revision: str = "main") -> WordEncoder:
    """Load a Hugging Face encoder that pools sub-words into word vectors.

    Raises:
        BackendUnavailableError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        msg = f"encoder {name!r} needs transformers; {SEMANTIC_EXTRA}"
        raise BackendUnavailableError(msg) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, add_prefix_space=True
    )
    return _WordPoolEncoder(
        tokenizer, AutoModel.from_pretrained(name, revision=revision).eval()
    )


class NliModel(Protocol):
    """Score entailment between premise and hypothesis."""

    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return ``P(premise entails hypothesis)`` for each pair."""


class _SequenceNli:
    def __init__(self, tokenizer: Any, model: Any) -> None:
        self._tokenizer = tokenizer
        self._model = model
        labels = {int(i): str(n).lower() for i, n in model.config.id2label.items()}
        self._entail = next(i for i, name in labels.items() if "entail" in name)

    def entailment(
        self, pairs: Sequence[tuple[str, str]]
    ) -> list[float]:  # pragma: no cover
        import torch

        batch = self._tokenizer(
            [p for p, _ in pairs],
            [h for _, h in pairs],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            probs = self._model(**batch).logits.float().softmax(dim=-1)
        return [float(v) for v in probs[:, self._entail].cpu().tolist()]


@lru_cache(maxsize=2)
def load_nli_model(name: str, revision: str = "main") -> NliModel:
    """Load a Hugging Face sequence-classification NLI model.

    Raises:
        BackendUnavailableError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        msg = f"model {name!r} needs transformers; {SEMANTIC_EXTRA}"
        raise BackendUnavailableError(msg) from exc
    return _SequenceNli(
        AutoTokenizer.from_pretrained(name, revision=revision),
        AutoModelForSequenceClassification.from_pretrained(
            name, revision=revision
        ).eval(),
    )


class SpeechEncoder(Protocol):
    """Map a waveform file to unit-length frame vectors."""

    def encode(self, path: str) -> NDArray[np.floating]:
        """Return an ``(n_frames, dim)`` array."""


class Recognizer(Protocol):
    """Transcribe a waveform file."""

    def transcribe(self, path: str) -> str:
        """Return the spoken words."""


def _read_waveform(path: str, rate: int) -> NDArray[np.float32]:
    from pathlib import Path

    from ..alignment.io import decode_audio

    return decode_audio(Path(path).read_bytes(), rate)[0]


class _WavLmEncoder:
    def __init__(self, extractor: Any, model: Any, layer: int) -> None:
        self._extractor = extractor
        self._model = model
        self._layer = layer

    def encode(self, path: str) -> NDArray[np.floating]:  # pragma: no cover
        import torch

        rate = int(self._extractor.sampling_rate)
        inputs = self._extractor(
            _read_waveform(path, rate), sampling_rate=rate, return_tensors="pt"
        )
        with torch.no_grad():
            hidden = self._model(**inputs, output_hidden_states=True).hidden_states[
                self._layer
            ]
        frames = torch.nn.functional.normalize(hidden[0].float(), dim=-1)
        return np.asarray(frames.cpu().numpy())


@lru_cache(maxsize=2)
def load_speech_encoder(name: str, layer: int, revision: str = "main") -> SpeechEncoder:
    """Load a self-supervised speech model and expose layer ``layer`` frames.

    Raises:
        BackendUnavailableError: If ``transformers`` is not installed.
    """
    try:
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        msg = f"encoder {name!r} needs transformers; {AUDIO_EXTRA}"
        raise BackendUnavailableError(msg) from exc
    extractor_class: Any = AutoFeatureExtractor
    return _WavLmEncoder(
        extractor_class.from_pretrained(name, revision=revision),
        AutoModel.from_pretrained(name, revision=revision).eval(),
        layer,
    )


class _PipelineRecognizer:
    def __init__(self, asr: Any) -> None:
        self._asr = asr

    def transcribe(self, path: str) -> str:
        rate = int(self._asr.feature_extractor.sampling_rate)
        audio = {"raw": _read_waveform(path, rate), "sampling_rate": rate}
        return str(self._asr(audio)["text"]).strip()


@lru_cache(maxsize=2)
def load_recognizer(name: str, revision: str = "main") -> Recognizer:
    """Load a Hugging Face speech-recognition pipeline.

    Raises:
        BackendUnavailableError: If ``transformers`` is not installed.
    """
    try:
        from transformers import pipeline
    except ImportError as exc:
        msg = f"recognizer {name!r} needs transformers; {AUDIO_EXTRA}"
        raise BackendUnavailableError(msg) from exc
    return _PipelineRecognizer(
        pipeline("automatic-speech-recognition", model=name, revision=revision)
    )
