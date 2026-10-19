"""Tests for the lazy encoder wrappers, with stub tokenizers and tiny torch models.

@see [[src/mllm_shap/value/encoders.py]]
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from mllm_shap.errors import BackendUnavailableError
from mllm_shap.value import encoders

torch = pytest.importorskip("torch")
soundfile = pytest.importorskip("soundfile")


@pytest.fixture(autouse=True)
def _fresh_loader_caches() -> "object":
    yield
    for name in dir(encoders):
        loader = getattr(encoders, name)
        if hasattr(loader, "cache_clear"):
            loader.cache_clear()


class _Batch(dict):
    """Tokenizer output: a dict that also answers ``word_ids``."""

    def __init__(self, *args: object, words: list[list[int | None]] | None = None):
        super().__init__(*args)
        self._words = words or []

    def word_ids(self, index: int) -> list[int | None]:
        return self._words[index]


class _Hidden:
    """Model stub returning fixed hidden states for any input."""

    def __init__(self, states: "torch.Tensor") -> None:
        self._states = states

    def __call__(self, **_: object) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            hidden_states=(self._states * 0, self._states),
            last_hidden_state=self._states,
        )


def test_hidden_state_encoder_drops_special_and_padding_tokens() -> None:
    def tokenizer(texts: list[str], **_: object) -> _Batch:
        return _Batch({
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1], [1, 0, 0, 0]]),
        })

    states = torch.arange(1.0, 25.0).reshape(2, 4, 3)
    encoder = encoders._HiddenStateEncoder(tokenizer, _Hidden(states), 1)
    first, second = encoder.encode(["a b", "c"])

    assert first.shape == (2, 3)  # two real tokens of four
    assert second.shape == (1, 3)  # one real token, the rest special or padding
    assert np.linalg.norm(first, axis=1) == pytest.approx(1.0)


def test_word_encoder_mean_pools_sub_words() -> None:
    def tokenizer(sentences: list[list[str]], **_: object) -> _Batch:
        return _Batch(
            {"input_ids": torch.zeros(1, 4, dtype=torch.long)}, words=[[None, 0, 0, 1]]
        )

    states = torch.tensor([[[9.0, 9.0], [2.0, 0.0], [4.0, 2.0], [1.0, 1.0]]])
    encoder = encoders._WordPoolEncoder(tokenizer, _Hidden(states))
    (vectors,) = encoder.encode([["ab", "c"]])

    assert vectors.tolist() == [[3.0, 1.0], [1.0, 1.0]]


def test_nli_model_returns_the_entailment_probability() -> None:
    class Model:
        config = types.SimpleNamespace(
            id2label={0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        )

        def __call__(self, **_: object) -> types.SimpleNamespace:
            return types.SimpleNamespace(logits=torch.tensor([[0.0, 0.0, 0.0]]))

    def tokenizer(*_: object, **__: object) -> dict[str, "torch.Tensor"]:
        return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

    nli = encoders._SequenceNli(tokenizer, Model())
    assert nli.entailment([("p", "h")]) == pytest.approx([1 / 3])


def test_speech_encoder_and_recognizer_read_a_wav_file(tmp_path: Path) -> None:
    path = tmp_path / "answer.wav"
    soundfile.write(path, np.zeros(1600, dtype=np.float32), 16000)

    class Extractor:
        sampling_rate = 16000

        def __call__(self, wave: np.ndarray, **_: object) -> dict[str, "torch.Tensor"]:
            assert len(wave) == 1600
            return {"input_values": torch.zeros(1, 4)}

    states = torch.ones(1, 5, 3)
    frames = encoders._WavLmEncoder(Extractor(), _Hidden(states), 1).encode(str(path))
    assert frames.shape == (5, 3)
    assert np.linalg.norm(frames, axis=1) == pytest.approx(1.0)

    class Asr:
        feature_extractor = Extractor()

        def __call__(self, audio: dict[str, object]) -> dict[str, str]:
            assert audio["sampling_rate"] == 16000
            return {"text": " hello "}

    assert encoders._PipelineRecognizer(Asr()).transcribe(str(path)) == "hello"


def test_sentence_encoder_wraps_the_library_model() -> None:
    class Model:
        def encode(self, texts: list[str], show_progress_bar: bool) -> list[list[int]]:
            assert not show_progress_bar
            return [[len(t)] for t in texts]

    wrapped = encoders._SentenceTransformer(Model())
    assert wrapped.encode(["ab", "c"]).tolist() == [[2], [1]]


@pytest.mark.parametrize(
    ("loader", "args", "module"),
    [
        (encoders.load_sentence_encoder, ("m",), "sentence_transformers"),
        (encoders.load_token_encoder, ("m", 1, "main"), "transformers"),
        (encoders.load_word_encoder, ("m", "main"), "transformers"),
        (encoders.load_nli_model, ("m", "main"), "transformers"),
        (encoders.load_speech_encoder, ("m", 1, "main"), "transformers"),
        (encoders.load_recognizer, ("m", "main"), "transformers"),
    ],
)
def test_loaders_name_the_extra_when_the_library_is_missing(
    monkeypatch: pytest.MonkeyPatch, loader: object, args: tuple, module: str
) -> None:
    monkeypatch.setitem(sys.modules, module, None)
    loader.cache_clear()  # type: ignore[attr-defined]
    with pytest.raises(BackendUnavailableError, match="pip install"):
        loader(*args)  # type: ignore[operator]


def test_loaders_build_wrappers_around_the_hub_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class Auto:
        def __init__(self, kind: str) -> None:
            self._kind = kind

        def from_pretrained(self, name: str, revision: str, **_: object) -> object:
            calls.append((self._kind, revision))
            stub = types.SimpleNamespace(
                config=types.SimpleNamespace(id2label={0: "entailment"}),
                eval=lambda: stub,
            )
            return stub

    fake = types.SimpleNamespace(
        AutoModel=Auto("model"),
        AutoTokenizer=Auto("tokenizer"),
        AutoFeatureExtractor=Auto("extractor"),
        AutoModelForSequenceClassification=Auto("classifier"),
        pipeline=lambda task, model, revision: calls.append((task, revision)),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=lambda name: object()),
    )
    for loader in (
        encoders.load_sentence_encoder,
        encoders.load_token_encoder,
        encoders.load_word_encoder,
        encoders.load_nli_model,
        encoders.load_speech_encoder,
        encoders.load_recognizer,
    ):
        loader.cache_clear()  # type: ignore[attr-defined]

    encoders.load_sentence_encoder("s")
    encoders.load_token_encoder("t", 3, "abc")
    encoders.load_word_encoder("w", "abc")
    encoders.load_nli_model("n", "abc")
    encoders.load_speech_encoder("sp", 3, "abc")
    encoders.load_recognizer("asr", "abc")

    assert ("model", "abc") in calls
    assert ("classifier", "abc") in calls
    assert ("extractor", "abc") in calls
    assert ("automatic-speech-recognition", "abc") in calls
