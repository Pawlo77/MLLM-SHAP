"""Test doubles shared by the pipeline tests."""

from mllm_shap.protocols.backend import GenerationResult


class WordCountValue:
    """v(S) = number of whitespace tokens in the coalition output."""

    def __init__(self, context: object | None = None) -> None:
        _ = context

    @property
    def name(self) -> str:
        return "word_count"

    def score(self, base: GenerationResult, candidate: GenerationResult) -> float:
        return float(len(candidate.text.split()))
