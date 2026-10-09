"""OpenAI-compatible HTTP generative backend.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_api.py]]
"""

import asyncio
import dataclasses
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlsplit

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...errors import BackendUnavailableError, ConfigError
from ...protocols.plugin import PluginOptions
from ...value.generation_records import generation_record_from_text
from .._chat import snapshot_has_audio, snapshot_to_chat_messages
from .payload import (
    build_chat_payload,
    extract_logprobs,
    extract_message_content,
    payload_cache_key,
)

if TYPE_CHECKING:
    from ...plugins.context import PluginContext

logger = logging.getLogger(__name__)

RESPONSE_CACHE_SIZE = 1024
"""Maximum deterministic responses kept in memory per backend instance."""

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
"""HTTP status codes treated as transient and retried."""

MAX_RETRY_DELAY_S = 8.0
"""Upper bound of the exponential backoff between retries."""

OPENAI_API_HOST = "api.openai.com"
"""The only host that receives ``OPENAI_API_KEY``."""


class ApiBackend:
    """Generate text through an OpenAI-compatible ``/chat/completions`` endpoint."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(
        self,
        context: "PluginContext",
        transport: Any | None = None,
    ) -> None:
        """Read ``context.config.backend``; ``transport`` overrides HTTP transport."""
        self._config = context.config.backend
        self._max_connections = context.config.explanation.max_inflight
        self._transport = transport
        self._client: Any | None = None
        self._response_cache: OrderedDict[str, GenerationRecord] = OrderedDict()
        self._init_lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        """Return the configured remote model identifier."""
        return self._config.model_id

    @property
    def returns_logprobs(self) -> bool:
        """Always true; requests ask for logprobs, servers that omit them give None."""
        return True

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        """Generate assistant text for ``snapshot`` via the HTTP API."""
        if snapshot_has_audio(snapshot):
            msg = "audio snapshots are not supported by the API backend"
            raise ValueError(msg)
        messages = snapshot_to_chat_messages(snapshot)
        payload = build_chat_payload(self._config.model_id, messages, generation)
        cache_key: str | None = None
        if generation.temperature == 0.0:
            cache_key = payload_cache_key(payload)
            cached = self._response_cache.get(cache_key)
            if cached is not None:
                self._response_cache.move_to_end(cache_key)
                return cached
        record = await self._post_chat_completions(payload)
        if cache_key is not None:
            self._response_cache[cache_key] = record
            while len(self._response_cache) > RESPONSE_CACHE_SIZE:
                self._response_cache.popitem(last=False)
        return record

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
        self._client = None

    async def _post_chat_completions(self, payload: dict[str, Any]) -> GenerationRecord:
        """POST ``payload``, retrying transient failures with exponential backoff.

        Status codes in ``RETRYABLE_STATUS`` and transport errors (timeouts,
        refused connections) are retried up to ``backend.max_retries`` times;
        the delay is ``Retry-After`` when the server sends it, capped at
        ``backend.timeout_s``, otherwise ``min(0.5 * 2**attempt, MAX_RETRY_DELAY_S)``
        seconds.

        Raises:
            BackendUnavailableError: If the request keeps failing, fails with a
                non-retryable status, or the response is not valid JSON.
        """
        client = await self._ensure_client()
        url = f"{self._base_url()}/chat/completions"
        headers = self._headers()
        httpx = _import_httpx()
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            retry_after: str | None = None
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status not in RETRYABLE_STATUS or attempt == attempts - 1:
                    msg = f"HTTP API generation request failed with status {status}"
                    raise BackendUnavailableError(msg) from exc
                retry_after = exc.response.headers.get("Retry-After")
            except httpx.TransportError as exc:
                if attempt == attempts - 1:
                    msg = "HTTP API generation request failed"
                    raise BackendUnavailableError(msg) from exc
            else:
                return _parse_response(response)
            delay = _retry_delay(attempt, retry_after, self._config.timeout_s)
            logger.debug("retrying HTTP API request in %.2f s", delay)
            await asyncio.sleep(delay)
        msg = "HTTP API generation request failed"
        raise BackendUnavailableError(msg)

    def _base_url(self) -> str:
        if not self._config.api_host:
            msg = "backend.api_host is required for the api backend"
            raise ConfigError(msg)
        return self._config.api_host.rstrip("/")

    def _headers(self) -> dict[str, str]:
        """Return request headers with the bearer key, when one applies.

        ``NLP_SHAP_API_KEY`` is sent to any host. ``OPENAI_API_KEY`` is sent only
        when it is the fallback and the host is ``api.openai.com``, so an OpenAI
        key never reaches a local or third-party server.
        """
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("NLP_SHAP_API_KEY")
        if not api_key and urlsplit(self._base_url()).hostname == OPENAI_API_HOST:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _ensure_client(self) -> Any:
        async with self._init_lock:
            if self._client is not None:
                return self._client
            httpx = _import_httpx()
            client_kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(self._config.timeout_s),
                "limits": httpx.Limits(max_connections=self._max_connections or 8),
            }
            if self._transport is not None:
                client_kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**client_kwargs)
            return self._client


def _parse_response(response: Any) -> GenerationRecord:
    try:
        data = response.json()
    except ValueError as exc:
        msg = "HTTP API response was not valid JSON"
        raise BackendUnavailableError(msg) from exc
    if not isinstance(data, dict):
        msg = "HTTP API response was not a JSON object"
        raise BackendUnavailableError(msg)
    record = generation_record_from_text(extract_message_content(data))
    return dataclasses.replace(record, logprobs=extract_logprobs(data))


def _retry_delay(attempt: int, retry_after: str | None, max_delay: float) -> float:
    if retry_after is not None:
        try:
            return min(max(0.0, float(retry_after)), max_delay)
        except ValueError:
            pass
    return min(0.5 * 2.0**attempt, MAX_RETRY_DELAY_S)


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        msg = "httpx package is required for the API backend extra"
        raise BackendUnavailableError(msg) from exc
    return httpx
