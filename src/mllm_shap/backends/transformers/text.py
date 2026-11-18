"""Hugging Face language model backend for text and, with a processor, audio.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_transformers_text.py]]
"""

import asyncio
import copy
import dataclasses
import itertools
import logging
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import ForcedScore, GenerationRecord
from ...errors import BackendUnavailableError
from ...protocols.plugin import PluginOptions
from ...runtime.kv_cache import CroppableCache, PrefixCacheManager, crop_to
from ...runtime.metrics import CostCounters
from ...value.generation_records import generation_record_from_text
from .._chat import snapshot_has_audio, snapshot_to_chat_messages
from ._torch import import_torch
from .audio import processor_inputs
from .chat import render_prompt
from .embeddings import contextual_embedding, static_embedding
from .options import TorchDtype, parse_transformers_options

if TYPE_CHECKING:
    from ...plugins.context import PluginContext

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ForceRequest:
    """One queued ``score_continuations`` call waiting for its batch."""

    snapshot: ConversationSnapshot
    continuations: tuple[str, ...]
    top_k: int
    future: "asyncio.Future[list[ForcedScore]]"


@dataclasses.dataclass
class _GenerateRequest:
    """One queued greedy ``generate`` call waiting for its batch."""

    snapshot: ConversationSnapshot
    generation: GenerationConfig
    future: "asyncio.Future[GenerationRecord]"


class TransformersTextBackend:
    """Generate text with a local Hugging Face language model.

    A checkpoint ``AutoModelForCausalLM`` cannot load (or any
    ``backend.options.model_class``) loads its own class with ``AutoProcessor``
    and also takes audio parts (see ``audio.processor_inputs``).
    """

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
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._prefix_cache: PrefixCacheManager | None = None
        self._kv_cache_hits = 0
        self._kv_cache_reused_tokens = 0
        self._cost = CostCounters()
        self._force_queue: list[_ForceRequest] = []
        self._generate_queue: list[_GenerateRequest] = []
        self._generate_timer: asyncio.Task[None] | None = None
        self._force_timer: asyncio.Task[None] | None = None
        self._warned_half_precision_cache = False
        self._init_lock = asyncio.Lock()
        # Generation runs in worker threads; one model must not run concurrently.
        self._model_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._config.model_id

    @property
    def batch_size(self) -> int:
        """Return the teacher-forced batch size (``backend.options.batch_size``)."""
        return self._options.batch_size

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
        """Generate assistant text for ``snapshot`` with optional prefix caching.

        Raises:
            ValueError: If ``snapshot`` has audio and the model has no processor.
        """
        await self._ensure_loaded()
        if snapshot_has_audio(snapshot) and self._processor is None:
            msg = (
                f"audio snapshots need an audio model; {self._config.model_id!r} "
                "loaded as a text-only causal language model"
            )
            raise ValueError(msg)
        if (
            self._options.batch_size > 1
            and generation.temperature == 0.0
            and not generation.compute_embeddings
            and not snapshot_has_audio(snapshot)
        ):
            return await self._enqueue_generate(snapshot, generation)
        return await asyncio.to_thread(
            self._generate_sync,
            snapshot,
            generation,
        )

    async def _enqueue_generate(
        self, snapshot: ConversationSnapshot, generation: GenerationConfig
    ) -> GenerationRecord:
        """Queue a greedy generation; run the batch when full or after the wait."""
        future: asyncio.Future[GenerationRecord] = (
            asyncio.get_running_loop().create_future()
        )
        self._generate_queue.append(_GenerateRequest(snapshot, generation, future))
        if len(self._generate_queue) >= self._options.batch_size:
            await self._flush_generate()
        elif self._generate_timer is None or self._generate_timer.done():
            self._generate_timer = asyncio.create_task(self._generate_after_wait())
        return await future

    async def _generate_after_wait(self) -> None:
        await asyncio.sleep(self._options.batch_wait_ms / 1000.0)
        await self._flush_generate()

    async def _flush_generate(self) -> None:
        batch, self._generate_queue = self._generate_queue, []
        if not batch:
            return
        try:
            records = await asyncio.to_thread(self._generate_batch_sync, batch)
        except Exception as error:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        for request, record in zip(batch, records, strict=True):
            if not request.future.done():
                request.future.set_result(record)

    def _generate_batch_sync(  # pragma: no cover
        self, batch: list["_GenerateRequest"]
    ) -> list[GenerationRecord]:
        """Generate every request greedily in left-padded batches.

        Requests with the same settings are sorted by prompt length and run
        ``batch_size`` at a time; the prefix cache is not used. Each row stops at its
        first end-of-sequence token, which is scored like in the one-prompt path.
        """
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            msg = "transformers backend is not loaded"
            raise BackendUnavailableError(msg)
        torch = import_torch()
        records: list[GenerationRecord | None] = [None] * len(batch)
        with self._model_lock, torch.no_grad():
            prompts = [
                [
                    int(t)
                    for t in self._encode(torch, model, tokenizer, r.snapshot)[
                        "input_ids"
                    ][0].tolist()
                ]
                for r in batch
            ]
            order = sorted(range(len(batch)), key=lambda i: len(prompts[i]))
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            eos = tokenizer.eos_token_id
            for start in range(0, len(order), self._options.batch_size):
                chunk = order[start : start + self._options.batch_size]
                for _key, members in itertools.groupby(
                    chunk,
                    key=lambda i: (
                        batch[i].generation.max_new_tokens,
                        batch[i].generation.top_k,
                    ),
                ):
                    group = list(members)
                    width = max(len(prompts[i]) for i in group)
                    device = model.device
                    ids = torch.tensor(
                        [[pad] * (width - len(prompts[i])) + prompts[i] for i in group],
                        dtype=torch.long,
                        device=device,
                    )
                    mask = torch.tensor(
                        [
                            [0] * (width - len(prompts[i])) + [1] * len(prompts[i])
                            for i in group
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    kwargs = _generate_kwargs(
                        torch, tokenizer, ids, batch[group[0]].generation
                    )
                    kwargs["attention_mask"] = mask
                    clock = _device_clock(torch)
                    generated = model.generate(**kwargs)
                    elapsed = _device_clock(torch) - clock
                    produced = 0
                    for row, i in enumerate(group):
                        new_ids = generated.sequences[row, width:].cpu().tolist()
                        if eos is not None and eos in new_ids:
                            new_ids = new_ids[: new_ids.index(eos) + 1]
                        steps = torch.stack([
                            step[row] for step in generated.logits[: len(new_ids)]
                        ]).float()
                        chosen = steps.log_softmax(dim=-1).gather(
                            1, torch.tensor(new_ids, device=steps.device)[:, None]
                        )
                        text = tokenizer.decode(
                            new_ids, skip_special_tokens=True
                        ).strip()
                        records[i] = dataclasses.replace(
                            generation_record_from_text(text),
                            logprobs=tuple(
                                float(v) for v in chosen[:, 0].cpu().tolist()
                            ),
                        )
                        produced += len(new_ids)
                    self._count(
                        sum(len(prompts[i]) for i in group),
                        produced,
                        len(group),
                        elapsed,
                    )
        return [record for record in records if record is not None]

    @property
    def forces_continuations(self) -> bool:
        """Always true for text prompts; audio prompts raise ``ValueError``."""
        return True

    async def score_continuations(  # pragma: no cover
        self,
        snapshot: ConversationSnapshot,
        continuations: Sequence[str],
        top_k: int = 0,
    ) -> list[ForcedScore]:
        """Score each continuation with one forward pass, without generating.

        The continuation is tokenized on its own and appended to the chat
        prompt; ``top_k`` adds the most probable next tokens per position.

        Raises:
            ValueError: If ``snapshot`` has audio parts.
        """
        await self._ensure_loaded()
        if snapshot_has_audio(snapshot):
            msg = "teacher-forced scoring supports text prompts only"
            raise ValueError(msg)
        if self._options.batch_size > 1:
            return await self._enqueue_force(snapshot, tuple(continuations), top_k)
        return await asyncio.to_thread(
            self._force_sync, snapshot, tuple(continuations), top_k
        )

    async def _enqueue_force(
        self, snapshot: ConversationSnapshot, continuations: tuple[str, ...], top_k: int
    ) -> list[ForcedScore]:
        """Queue a request; run the batch when full or after ``batch_wait_ms``."""
        future: asyncio.Future[list[ForcedScore]] = (
            asyncio.get_running_loop().create_future()
        )
        self._force_queue.append(_ForceRequest(snapshot, continuations, top_k, future))
        queued = sum(len(r.continuations) for r in self._force_queue)
        if queued >= self._options.batch_size:
            await self._flush_force()
        elif self._force_timer is None or self._force_timer.done():
            self._force_timer = asyncio.create_task(self._flush_after_wait())
        return await future

    async def _flush_after_wait(self) -> None:
        await asyncio.sleep(self._options.batch_wait_ms / 1000.0)
        await self._flush_force()

    async def _flush_force(self) -> None:
        batch, self._force_queue = self._force_queue, []
        if not batch:
            return
        try:
            results = await asyncio.to_thread(self._force_batch_sync, batch)
        except Exception as error:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(error)
            return
        for request, scores in zip(batch, results, strict=True):
            if not request.future.done():
                request.future.set_result(scores)

    def _force_batch_sync(  # pragma: no cover
        self, batch: list["_ForceRequest"]
    ) -> list[list[ForcedScore]]:
        """Score every continuation of ``batch`` in padded forward passes.

        Sequences are right-padded and sorted by length, ``batch_size`` per pass, so
        the scores equal the one-prompt passes up to floating-point rounding.
        """
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            msg = "transformers backend is not loaded"
            raise BackendUnavailableError(msg)
        torch = import_torch()
        jobs: list[tuple[int, int, list[int], list[int], int]] = []
        with self._model_lock, torch.no_grad():
            for r_index, request in enumerate(batch):
                prompt = [
                    int(t)
                    for t in self._encode(torch, model, tokenizer, request.snapshot)[
                        "input_ids"
                    ][0].tolist()
                ]
                for c_index, text in enumerate(request.continuations):
                    ids = [
                        int(t) for t in tokenizer.encode(text, add_special_tokens=False)
                    ]
                    jobs.append((r_index, c_index, prompt, ids, request.top_k))
            results: list[list[ForcedScore]] = [
                [ForcedScore(())] * len(r.continuations) for r in batch
            ]
            pending = sorted(
                (job for job in jobs if job[3]), key=lambda j: len(j[2]) + len(j[3])
            )
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            size = self._options.batch_size
            for start in range(0, len(pending), size):
                group = pending[start : start + size]
                width = max(len(j[2]) + len(j[3]) for j in group)
                rows = [
                    j[2] + j[3] + [pad] * (width - len(j[2]) - len(j[3])) for j in group
                ]
                mask = [
                    [1] * (len(j[2]) + len(j[3]))
                    + [0] * (width - len(j[2]) - len(j[3]))
                    for j in group
                ]
                device = model.device
                clock = _device_clock(torch)
                logits = model(
                    input_ids=torch.tensor(rows, dtype=torch.long, device=device),
                    attention_mask=torch.tensor(mask, dtype=torch.long, device=device),
                ).logits.float()
                elapsed = _device_clock(torch) - clock
                for row, (r_index, c_index, prompt, ids, top_k) in enumerate(group):
                    begin = len(prompt) - 1
                    results[r_index][c_index] = _forced_from_logits(
                        torch, logits[row, begin : begin + len(ids)], ids, top_k
                    )
                self._count(
                    sum(len(j[2]) + len(j[3]) for j in group), 0, len(group), elapsed
                )
        return results

    async def aclose(self) -> None:
        """Release loaded model references."""
        self._tokenizer = None
        self._processor = None
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
            model_id, revision = self._config.model_id, self._options.revision
            load_kwargs: dict[str, Any] = {"revision": revision, "dtype": dtype}
            if self._options.attn_implementation is not None:
                load_kwargs["attn_implementation"] = self._options.attn_implementation
            processor = None
            try:
                config = transformers.AutoConfig.from_pretrained(
                    model_id, revision=revision
                )
                model_class = self._options.model_class
                if model_class is None and (
                    type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING
                ):
                    tokenizer = transformers.AutoTokenizer.from_pretrained(
                        model_id, revision=revision
                    )
                    model_type = transformers.AutoModelForCausalLM
                else:
                    processor = transformers.AutoProcessor.from_pretrained(
                        model_id, revision=revision
                    )
                    tokenizer = processor.tokenizer
                    model_type = getattr(
                        transformers, model_class or config.architectures[0]
                    )
                model = model_type.from_pretrained(model_id, **load_kwargs)
            except Exception as exc:
                msg = f"failed to load transformers model {model_id!r}"
                raise BackendUnavailableError(msg) from exc
            model.to(device)
            model.eval()
            _ensure_pad_token(tokenizer)
            self._tokenizer = tokenizer
            self._processor = processor
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
        torch = import_torch()
        # Fast tokenizers are not thread-safe ("Already borrowed"), so tokenizing
        # and decoding share the model lock with generation.
        with self._model_lock, torch.no_grad():
            inputs = self._encode(torch, model, tokenizer, snapshot)
            input_ids = inputs["input_ids"]
            generate_kwargs = _generate_kwargs(torch, tokenizer, input_ids, generation)
            generate_kwargs.update(inputs)
            if len(inputs) == 1:
                token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
                self._attach_prefix_cache(model, input_ids, token_ids, generate_kwargs)
            started = _device_clock(torch)
            generated = model.generate(**generate_kwargs)
            elapsed = _device_clock(torch) - started
            new_ids = generated.sequences[0, input_ids.shape[1] :]
            new_token_ids = [int(token_id) for token_id in new_ids.cpu().tolist()]
            self._count(int(input_ids.shape[1]), len(new_token_ids), 1, elapsed)
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

    def _force_sync(  # pragma: no cover
        self, snapshot: ConversationSnapshot, continuations: tuple[str, ...], top_k: int
    ) -> list[ForcedScore]:
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            msg = "transformers backend is not loaded"
            raise BackendUnavailableError(msg)
        torch = import_torch()
        with self._model_lock, torch.no_grad():
            prompt_ids = self._encode(torch, model, tokenizer, snapshot)["input_ids"]
            scores = []
            for text in continuations:
                started = _device_clock(torch)
                scores.append(
                    _force_one(torch, model, prompt_ids, tokenizer, text, top_k)
                )
                elapsed = _device_clock(torch) - started
                forced = len(tokenizer.encode(text, add_special_tokens=False))
                self._count(int(prompt_ids.shape[1]) + forced, 0, 1, elapsed)
            return scores

    @property
    def cost_counters(self) -> CostCounters:
        """Return the cumulative model work since the backend was created."""
        return self._cost

    def _count(
        self, prompt: int, generated: int, sequences: int, seconds: float
    ) -> None:
        self._cost = CostCounters(
            self._cost.forward_sequences + sequences,
            self._cost.prompt_tokens + prompt,
            self._cost.generated_tokens + generated,
            self._cost.device_seconds + seconds,
        )

    def _encode(
        self, torch: Any, model: Any, tokenizer: Any, snapshot: ConversationSnapshot
    ) -> dict[str, Any]:
        """Return the model inputs; only text prompts give ``input_ids`` alone.

        Audio prompts carry processor features (cast to the model's dtype) and
        skip the prefix cache: cached prompt tokens do not cover the audio.
        """
        if self._processor is not None and snapshot_has_audio(snapshot):
            encoded = processor_inputs(self._processor, snapshot)
            return {
                key: value.to(self._device, dtype=model.dtype)
                if value.is_floating_point()
                else value.to(self._device)
                for key, value in dict(encoded).items()
            }
        prompt = render_prompt(tokenizer, snapshot_to_chat_messages(snapshot))
        token_ids = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=True)]
        if not token_ids:
            msg = "transformers backend received an empty prompt"
            raise BackendUnavailableError(msg)
        return {
            "input_ids": torch.tensor(
                [token_ids], dtype=torch.long, device=self._device
            )
        }

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
                crop_to(past, matched)
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


def _device_clock(torch: Any) -> float:  # pragma: no cover
    """Return a clock reading after pending device work has finished."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    return time.perf_counter()


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


def _force_one(  # pragma: no cover
    torch: Any, model: Any, prompt_ids: Any, tokenizer: Any, text: str, top_k: int
) -> ForcedScore:
    """Return the log probabilities of ``text`` after ``prompt_ids``, one pass."""
    ids = [int(t) for t in tokenizer.encode(text, add_special_tokens=False)]
    if not ids:
        return ForcedScore(())
    cont = torch.tensor([ids], dtype=torch.long, device=prompt_ids.device)
    full = torch.cat([prompt_ids, cont], dim=1)
    start = prompt_ids.shape[1] - 1
    logits = model(full).logits[0, start : start + len(ids)].float()
    return _forced_from_logits(torch, logits, ids, top_k)


def _forced_from_logits(  # pragma: no cover
    torch: Any, logits: Any, ids: list[int], top_k: int
) -> ForcedScore:
    """Return the forced score of ``ids`` from the logits at their positions."""
    cont = torch.tensor([ids], dtype=torch.long, device=logits.device)
    logprobs = logits.log_softmax(dim=-1)
    chosen = logprobs.gather(1, cont[0][:, None])[:, 0]
    token_logprobs = tuple(float(v) for v in chosen.cpu().tolist())
    if top_k <= 0:
        return ForcedScore(token_logprobs)
    top = logprobs.topk(min(top_k, logprobs.shape[-1]), dim=-1)
    return ForcedScore(
        token_logprobs,
        tuple(tuple(int(i) for i in row) for row in top.indices.cpu().tolist()),
        tuple(tuple(float(v) for v in row) for row in top.values.cpu().tolist()),
    )


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
