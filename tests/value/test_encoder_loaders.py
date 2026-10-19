"""Tests for the lazy encoder loaders; they need neither torch nor transformers.

@see [[src/mllm_shap/value/encoders.py]]
"""

import sys
import types

import pytest

from mllm_shap.errors import BackendUnavailableError
from mllm_shap.value import encoders


@pytest.fixture(autouse=True)
def _fresh_loader_caches() -> "object":
    yield
    for name in dir(encoders):
        loader = getattr(encoders, name)
        if hasattr(loader, "cache_clear"):
            loader.cache_clear()


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
