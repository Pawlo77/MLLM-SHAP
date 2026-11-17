"""Infill absence policy: absent words are replaced by a masked language model's fill.

Each absent player becomes one mask token per word. The model then fills every mask of
a message in one forward pass, taking the most probable word at each position, so the
same coalition always renders the same text. Fills are cached per masked text.

@domain masking
@see [[knowledge/architecture/masking]]
@tested-by [[tests/masking/test_infill.py]]
"""

import re
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from ..domain.coalition import CoalitionMask
from ..domain.conversation import ConversationSnapshot, Message, Turn
from ..domain.players import PlayerSet
from ..errors import BackendUnavailableError
from ..protocols.plugin import PluginOptions
from .policies import _render_audio, _require_spans
from .tokens import render_spans

if TYPE_CHECKING:
    from ..plugins.context import PluginContext


class MaskFiller(Protocol):
    """Fill the mask tokens of a text with words."""

    @property
    def mask_token(self) -> str:
        """Return the token that marks a word to fill."""

    def fill(self, text: str) -> list[str]:
        """Return one word per occurrence of :attr:`mask_token`, in order."""


class InfillOptions(PluginOptions):
    """Options of the ``infill`` absence policy."""

    model: str = "roberta-base"
    """Hugging Face masked language model that supplies the fills."""

    revision: str | None = None
    """Hub revision of the model; pin a commit hash for reproducible fills."""


class HfMaskedLmFiller:
    """Fill masks with the top word of a Hugging Face masked language model."""

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        """Remember ``model_id`` and ``revision``; the model loads on first use."""
        self._model_id = model_id
        self._revision = revision
        self._tokenizer: object | None = None
        self._model: object | None = None
        self._torch: Any = None

    @property
    def mask_token(self) -> str:
        """Return the model tokenizer's mask token."""
        return str(self._load()[0].mask_token)

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as error:
                msg = (
                    "absence policy 'infill' needs transformers; install it with "
                    "'mllm-shap[infill]'"
                )
                raise BackendUnavailableError(msg) from error
            self._torch = torch  # pragma: no cover
            self._tokenizer = AutoTokenizer.from_pretrained(  # pragma: no cover
                self._model_id, revision=self._revision
            )
            self._model = AutoModelForMaskedLM.from_pretrained(  # pragma: no cover
                self._model_id, revision=self._revision
            ).eval()
        return self._tokenizer, self._model

    def fill(self, text: str) -> list[str]:  # pragma: no cover
        """Return the most probable non-special word at each mask, in one pass."""
        tokenizer, model = self._load()
        torch = self._torch
        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        banned = torch.tensor(tokenizer.all_special_ids)
        words = []
        for position in positions[:, 0].tolist():
            scores = logits[position].clone()
            scores[banned] = float("-inf")
            words.append(tokenizer.decode([int(scores.argmax())]).strip())
        return words


class InfillPolicy:
    """Replace absent players with the fills of a masked language model."""

    options_model: ClassVar[type[PluginOptions]] = InfillOptions
    """Options accepted from ``explanation.plugin_options``."""

    def __init__(
        self,
        context: "PluginContext | None" = None,
        filler: MaskFiller | None = None,
    ) -> None:
        """Use ``filler`` or the model of options ``model`` and ``revision``."""
        options = InfillOptions.model_validate(
            {} if context is None else dict(context.options)
        )
        self._filler: MaskFiller = filler or HfMaskedLmFiller(
            options.model, options.revision
        )
        self._cache: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        """Return the registered absence-policy identifier."""
        return "infill"

    def apply(
        self,
        snapshot: ConversationSnapshot,
        players: PlayerSet,
        mask: CoalitionMask,
    ) -> ConversationSnapshot:
        """Fill absent words with the model's choice; silence absent audio."""
        spans = _require_spans(players, mask)
        token = self._filler.mask_token
        rendered = render_spans(
            snapshot,
            spans,
            [
                span.text
                if present
                else " ".join([token] * max(1, len(span.text.split())))
                for span, present in zip(spans, mask.present, strict=True)
            ],
        )
        filled = ConversationSnapshot.from_turns(
            tuple(
                Turn(
                    messages=tuple(self._fill_message(m, token) for m in turn.messages)
                )
                for turn in rendered.turns
            )
        )
        return _render_audio(filled, players, mask, cut=False)

    def _fill_message(self, message: Message, token: str) -> Message:
        text = message.text
        if token not in text:
            return message
        if text not in self._cache:
            self._cache[text] = self._filler.fill(text)
        words = iter(self._cache[text])
        return message.with_text(
            re.sub(re.escape(token), lambda _match: next(words), text)
        )
