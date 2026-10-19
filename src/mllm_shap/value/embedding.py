"""Embedding-cosine value function.

@domain value
@see [[knowledge/architecture/value]]
@tested-by [[tests/value/test_embedding.py]]
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..domain.enums import EmbeddingMode
from ..domain.generation import GenerationRecord
from ..protocols.backend import GenerationResult
from ..protocols.embedding import EmbeddingProvider
from ..protocols.plugin import PluginOptions
from ..protocols.value import ValueInfo
from ._metrics import cosine_similarity
from .encoders import SentenceEncoder, load_sentence_encoder

if TYPE_CHECKING:
    from ..plugins.context import PluginContext

MODEL_STATES = "model_states"
"""``encoder`` value that scores the explained model's own pooled states."""

DEFAULT_ENCODER = "all-MiniLM-L6-v2"
"""Sentence-transformers model scored by default, as in MExGen and llmSHAP."""


class EmbeddingOptions(PluginOptions):
    """Options of the ``embedding_cosine`` value function."""

    encoder: str = DEFAULT_ENCODER
    """Sentence-transformers model name, or ``model_states`` for the pooled input
    or last-layer states of the explained model (the pre-0.7 behaviour)."""

    allow_hash_embeddings: bool = False
    """Let ``model_states`` score the 8-dimensional hash vectors of backends
    without model states (mock, LM Studio, API); they carry no meaning."""


class CosineEmbeddingValue:
    """Cosine similarity between sentence embeddings of two answers.

    With a sentence ``encoder`` (default ``all-MiniLM-L6-v2``, extra
    ``mllm-shap[semantic]``) both answers are embedded from their text, so
    ``explain`` and ``reanalyze`` agree. With ``encoder: model_states`` it
    compares the static or contextual embeddings the backend attaches to each
    record (``explanation.embedding_mode``); built without a ``context`` it
    defaults to that mode, so records carry their own vectors.
    """

    options_model: ClassVar[type[PluginOptions]] = EmbeddingOptions
    """Options accepted from ``explanation.plugin_options``."""

    registered_name: ClassVar[str] = "embedding_cosine"
    """Registered value-function identifier."""

    info: ClassVar[ValueInfo] = ValueInfo(
        value_range=(-1.0, 1.0),
        reads_base=True,
        needs=("embeddings",),
        extra_model="sentence-transformers encoder (default all-MiniLM-L6-v2)",
        cost="encoder",
    )

    def __init__(
        self,
        context: "PluginContext | None" = None,
        embedding_mode: EmbeddingMode = EmbeddingMode.STATIC,
        provider: EmbeddingProvider | None = None,
        encoder: SentenceEncoder | None = None,
    ) -> None:
        """Pick the encoder and mode; ``encoder`` injects a loaded sentence encoder."""
        options = EmbeddingOptions(encoder=MODEL_STATES)
        if context is not None:
            options = EmbeddingOptions.model_validate(context.options)
            embedding_mode = context.config.explanation.embedding_mode
        self._embedding_mode = embedding_mode
        self._provider = provider
        self._encoder_name = options.encoder
        self.allow_hash_embeddings = options.allow_hash_embeddings
        """Whether hash vectors of a backend without model states may be scored."""
        self._encoder = encoder
        self._cache: dict[str, np.ndarray] = {}

    @property
    def name(self) -> str:
        """Return the registered value-function identifier."""
        return self.registered_name

    @property
    def needs_embeddings(self) -> bool:
        """True only for ``encoder: model_states``, which reads backend vectors."""
        return self._encoder_name == MODEL_STATES

    def manifest_info(self) -> dict[str, str]:
        """Return the encoder and its library version for the run manifest."""
        if self._encoder_name == MODEL_STATES:
            return {"encoder": MODEL_STATES}
        try:
            library = version("sentence-transformers")
        except PackageNotFoundError:
            library = "unknown"
        return {"encoder": self._encoder_name, "sentence_transformers": library}

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        """Return the cosine similarity of the two embedding vectors."""
        if self._encoder_name == MODEL_STATES:
            return cosine_similarity(
                _resolve_embedding(base, self._embedding_mode, self._provider),
                _resolve_embedding(candidate, self._embedding_mode, self._provider),
            )
        return cosine_similarity(self._embed(base.text), self._embed(candidate.text))

    def _embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            if self._encoder is None:
                self._encoder = load_sentence_encoder(self._encoder_name)
            self._cache[text] = self._encoder.encode([text])[0]
        return self._cache[text]


def _resolve_embedding(
    generation: GenerationResult,
    embedding_mode: EmbeddingMode,
    provider: EmbeddingProvider | None,
) -> tuple[float, ...]:
    if isinstance(generation, GenerationRecord):
        if embedding_mode == EmbeddingMode.CONTEXTUAL:
            if generation.contextual_embedding:
                return generation.contextual_embedding
            if generation.embedding:
                return generation.embedding
        elif generation.embedding:
            return generation.embedding
    if provider is not None:
        return provider.embed(generation)
    msg = "embedding vector or EmbeddingProvider is required"
    raise TypeError(msg)
