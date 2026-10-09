"""Hugging Face causal language model backend (text-only).

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_transformers_text.py]]
"""

import asyncio
import copy
import dataclasses
import logging
import threading
from typing import TYPE_CHECKING, Any, ClassVar

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...errors import BackendUnavailableError
from ...protocols.plugin import PluginOptions
from ...runtime.kv_cache import CroppableCache, PrefixCacheManager
from ...value.generation_records import generation_record_from_text
from .._chat import snapshot_has_audio, snapshot_to_chat_messages
from ._torch import import_torch
from .chat import render_prompt
from .embeddings import contextual_embedding, static_embedding
from .options import TorchDtype, parse_transformers_options

if TYPE_CHECKING:
    from ...plugins.context import PluginContext

logger = logging.getLogger(__name__)


class TransformersTextBackend:
    """Generate text with a local Hugging Face causal language model."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext") -> None:
        """Read ``context.config.backend``; the model loads on first use.

        Raises:
            ConfigError: If ``backend.options`` holds an unknown key or an
                unsupported ``dtype``.
        """
        self._config = context.config.backend
        self._options = parse_transformers_options(self._config.options)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._prefix_cache: PrefixCacheManager | None = None
        self._kv_cache_hits = 0
        self._kv_cache_reused_tokens = 0
        self._warned_half_precision_cache = False
        self._init_lock = asyncio.Lock()
        # Generation runs in worker threads; one model must not run concurrently.
        self._model_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._config.model_id

    @property
    def returns_logprobs(self) -> bool:
        """Always true; every record carries its generated tokens' logprobs."""
        return True

    @property
    def dtype(self) -> TorchDtype:
        """Return the configured weight dtype (``backend.options.dtype``)."""
        return self._options.dtype

    @property
    def kv_cache_hits(self) -> int:
        """Return the number of prefix-cache hits during this backend lifetime."""
        return self._kv_cache_hits

    @property
    def kv_cache_reused_tokens(self) -> int:
        """Return the prompt tokens served from cached prefixes, summed over hits."""
        return self._kv_cache_reused_tokens

    def set_kv_cache_enabled(self, enabled: bool, max_entries: int) -> None:
        """Enable or disable prefix-cache reuse, keeping at most ``max_entries``."""
        if enabled:
            self._prefix_cache = PrefixCacheManager(max_entries=max_entries)
        else:
            self._prefix_cache = None
            self._kv_cache_hits = 0
            self._kv_cache_reused_tokens = 0

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        """Generate assistant text for ``snapshot`` with optional prefix caching."""
        if snapshot_has_audio(snapshot):
            msg = "audio snapshots are not supported by the transformers text backend"
            raise ValueError(msg)
        await self._ensure_loaded()
        return await asyncio.to_thread(
            self._generate_sync,
            snapshot,
            generation,
        )

    async def aclose(self) -> None:
        """Release loaded model references."""
        self._tokenizer = None
        self._model = None
        self._device = None
        if self._prefix_cache is not None:
            self._prefix_cache.reset()

    async def _ensure_loaded(self) -> None:
        async with self._init_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            try:
                torch = import_torch()
                transformers = _import_transformers()
            except ImportError as exc:
                msg = "transformers backend requires the transformers optional extra"
                raise BackendUnavailableError(msg) from exc
            device, dtype = _device_and_dtype(torch, self._options.dtype)
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self._config.model_id,
                    revision=self._options.revision,
                )
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self._config.model_id,
                    revision=self._options.revision,
                    dtype=dtype,
                )
            except Exception as exc:
                msg = f"failed to load transformers model {self._config.model_id!r}"
                raise BackendUnavailableError(msg) from exc
            model.to(device)
            model.eval()
            _ensure_pad_token(tokenizer)
            self._tokenizer = tokenizer
            self._model = model
            self._device = device

    def _generate_sync(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            msg = "transformers backend is not loaded"
            raise BackendUnavailableError(msg)
        messages = snapshot_to_chat_messages(snapshot)
        torch = import_torch()
        # Fast tokenizers are not thread-safe ("Already borrowed"), so tokenizing
        # and decoding share the model lock with generation.
        with self._model_lock, torch.no_grad():
            prompt = render_prompt(tokenizer, messages)
            encoded = tokenizer.encode(prompt, add_special_tokens=True)
            token_ids = [int(token_id) for token_id in encoded]
            if not token_ids:
                msg = "transformers backend received an empty prompt"
                raise BackendUnavailableError(msg)
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
            generate_kwargs = _generate_kwargs(torch, tokenizer, input_ids, generation)
            self._attach_prefix_cache(model, input_ids, token_ids, generate_kwargs)
            generated = model.generate(**generate_kwargs)
            new_ids = generated.sequences[0, len(token_ids) :]
            new_token_ids = [int(token_id) for token_id in new_ids.cpu().tolist()]
            text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
            record = dataclasses.replace(
                generation_record_from_text(text),
                logprobs=_token_logprobs(torch, generated.logits, new_ids),
            )
            if not generation.compute_embeddings:
                return record
            return dataclasses.replace(
                record,
                embedding=static_embedding(model, new_token_ids),
                contextual_embedding=contextual_embedding(model, new_token_ids),
            )

    def _attach_prefix_cache(
        self,
        model: Any,
        input_ids: Any,
        token_ids: list[int],
        generate_kwargs: dict[str, Any],
    ) -> None:
        """Generate from a cached prompt prefix, caching new prefixes on the way.

        Coalitions with the same players up to the last one share a prompt
        prefix (every token but the last). The longest cached match is copied
        and cropped (``generate`` mutates the state it receives); the prefix
        tokens it does not cover are run through the model and the completed
        prefix state is stored. On a miss that is the whole prefix. In both
        cases ``generate`` continues from the computed state, so no prefix token
        is processed twice.
        """
        cache = self._prefix_cache
        prefix_len = len(token_ids) - 1
        if cache is None or prefix_len < 1:
            return
        self._warn_half_precision_cache(model)
        cached, matched = cache.longest_prefix(token_ids)
        past = None
        if matched:
            past = copy.deepcopy(cached)
            if isinstance(past, CroppableCache):
                past.crop(matched)
            self._kv_cache_hits += 1
            self._kv_cache_reused_tokens += matched
        if matched < prefix_len:
            out = model(
                input_ids[:, matched:prefix_len],
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            cache.store(token_ids[:prefix_len], copy.deepcopy(past))
        generate_kwargs["past_key_values"] = past

    def _warn_half_precision_cache(self, model: Any) -> None:
        """Warn once that a half-precision prefix cache is not bit-identical (D04).

        The cached path runs the prompt in several forward passes and the uncached
        path in one; float16 and bfloat16 round those differently, which can flip a
        near-tied greedy token. float32 keeps both paths identical.
        """
        dtype = str(getattr(model, "dtype", ""))
        if self._warned_half_precision_cache or not dtype.endswith((
            "float16",
            "bfloat16",
        )):
            return
        self._warned_half_precision_cache = True
        logger.warning(
            "kv_cache with %s weights: cached prefixes round differently from one "
            "forward pass, so greedy outputs and values can differ slightly from "
            "kv_cache off; set backend.options.dtype: float32 for identical results",
            dtype.removeprefix("torch."),
        )


def _generate_kwargs(
    torch: Any, tokenizer: Any, input_ids: Any, generation: GenerationConfig
) -> dict[str, Any]:
    """Return ``model.generate`` arguments; raw logits come back for logprobs."""
    do_sample = generation.temperature > 0.0
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": generation.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    if do_sample:
        kwargs["temperature"] = generation.temperature
        if generation.top_k > 0:
            kwargs["top_k"] = generation.top_k
    return kwargs


def _token_logprobs(torch: Any, logits: Any, new_ids: Any) -> tuple[float, ...]:
    """Return the log-softmax of the raw logits at each generated token.

    ``logits`` holds one ``(1, vocab)`` tensor per generation step, before
    temperature or top-k warping, so the values are the model's own
    probabilities; the end-of-sequence token is scored when it was generated.
    """
    if len(new_ids) == 0:
        return ()
    steps = torch.stack([step[0] for step in logits[: len(new_ids)]]).float()
    chosen = steps.log_softmax(dim=-1).gather(1, new_ids[:, None].to(steps.device))
    return tuple(float(value) for value in chosen[:, 0].cpu().tolist())


def _device_and_dtype(torch: Any, dtype: TorchDtype) -> tuple[Any, Any]:
    """Pick CUDA, then Apple MPS, then CPU, and the weight dtype for it.

    ``auto`` prefers bfloat16 on accelerators that support it (float16 overflows
    to NaN on many bfloat16-trained models), falls back to float16 on other
    accelerators and uses float32 on CPU.
    """
    if torch.cuda.is_available():
        device, bf16 = torch.device("cuda"), bool(torch.cuda.is_bf16_supported())
    elif torch.backends.mps.is_available():
        device, bf16 = torch.device("mps"), _mps_supports_bfloat16(torch)
    else:
        device, bf16 = torch.device("cpu"), False
    if dtype != "auto":
        return device, getattr(torch, dtype)
    if device == torch.device("cpu"):
        return device, torch.float32
    return device, torch.bfloat16 if bf16 else torch.float16


def _mps_supports_bfloat16(torch: Any) -> bool:
    try:
        torch.ones(1, dtype=torch.bfloat16, device="mps")
    except (RuntimeError, TypeError):
        return False
    return True


def _ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        msg = "transformers is required for the transformers backend extra"
        raise ImportError(msg) from exc
    return transformers
