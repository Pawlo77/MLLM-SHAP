"""LM Studio generative backend.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_lmstudio.py]]
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from ...domain.config import GenerationConfig
from ...domain.conversation import ConversationSnapshot
from ...domain.generation import GenerationRecord
from ...errors import BackendUnavailableError
from ...protocols.plugin import PluginOptions
from ...value.generation_records import generation_record_from_text
from .._chat import snapshot_has_audio
from .chat import snapshot_to_chat
from .models import resolve_model_key
from .options import LmStudioOptions, parse_lmstudio_options

if TYPE_CHECKING:
    from ...plugins.context import PluginContext


class LmStudioBackend:
    """Generate text through a local LM Studio API server."""

    options_model: ClassVar[type[PluginOptions]] = PluginOptions
    """Accepts no ``explanation.plugin_options``."""

    def __init__(self, context: "PluginContext") -> None:
        """Read ``context.config.backend`` and validate its LM Studio ``options``."""
        self._config = context.config.backend
        self._options = parse_lmstudio_options(self._config.options)
        self._client: Any | None = None
        self._model: Any | None = None
        self._init_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()

    @property
    def options(self) -> LmStudioOptions:
        """Return the validated LM Studio options."""
        return self._options

    @property
    def model_id(self) -> str:
        """Return the configured LM Studio model identifier."""
        return self._config.model_id

    async def generate(
        self,
        snapshot: ConversationSnapshot,
        generation: GenerationConfig,
    ) -> GenerationRecord:
        """Generate assistant text for ``snapshot`` via LM Studio."""
        if snapshot_has_audio(snapshot):
            msg = "audio snapshots are not supported by the LM Studio backend"
            raise ValueError(msg)
        lms = _import_lmstudio()
        chat = snapshot_to_chat(snapshot, lms.Chat)
        prediction_config: dict[str, float | int] = {
            "temperature": generation.temperature,
            "maxTokens": generation.max_new_tokens,
        }
        if generation.top_k > 0:
            prediction_config["topKSampling"] = generation.top_k
        async with self._generation_slot():
            model = await self._ensure_model()
            try:
                result = await model.respond(chat, config=prediction_config)
            except (ValueError, TypeError):
                raise
            except Exception as exc:
                if not _is_transient(exc):
                    msg = "LM Studio generation request failed"
                    raise BackendUnavailableError(msg) from exc
                # The connection dropped or the model unloaded: reload, retry once.
                self._model = None
                try:
                    model = await self._ensure_model()
                    result = await model.respond(chat, config=prediction_config)
                except Exception as retry_exc:
                    msg = "LM Studio generation request failed"
                    raise BackendUnavailableError(msg) from retry_exc
            text = _extract_text(result)
            return generation_record_from_text(text)

    async def aclose(self) -> None:
        """Close the underlying LM Studio client connection."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
        self._client = None
        self._model = None

    @asynccontextmanager
    async def _generation_slot(self) -> AsyncGenerator[None]:
        if self._options.serialize_generate:
            async with self._generate_lock:
                yield
        else:
            yield

    async def _ensure_model(self) -> Any:
        async with self._init_lock:
            if self._model is not None:
                return self._model
            client = await self._connect_client()
            try:
                model_key = await resolve_model_key(
                    client, self._config.model_id, self._options
                )
                # ttl=None keeps the instance loaded (UI Idle TTL still applies).
                self._model = await client.llm.model(model_key, ttl=None)
            except BackendUnavailableError:
                await self.aclose()
                raise
            except Exception as exc:
                await self.aclose()
                msg = f"failed to load LM Studio model {self._config.model_id!r}"
                raise BackendUnavailableError(msg) from exc
            return self._model

    async def _connect_client(self) -> Any:
        if self._client is not None:
            return self._client
        lms = _import_lmstudio()
        host = self._config.api_host
        if not host:
            host = await lms.AsyncClient.find_default_local_api_host()
        if host is None or not await lms.AsyncClient.is_valid_api_host(host):
            msg = "LM Studio API server is not reachable on the local machine"
            raise BackendUnavailableError(msg)
        client = lms.AsyncClient(host)
        try:
            await client.__aenter__()
        except Exception as exc:
            msg = f"failed to connect to LM Studio API host {host!r}"
            raise BackendUnavailableError(msg) from exc
        self._client = client
        return client


def _is_transient(error: Exception) -> bool:
    """Return whether ``error`` looks like a dropped connection worth one retry."""
    if isinstance(error, OSError):
        return True
    return any(
        "Websocket" in cls.__name__ or "Connection" in cls.__name__
        for cls in type(error).__mro__
    )


def _extract_text(result: object) -> str:
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content.strip()
    msg = "LM Studio prediction result did not include text content"
    raise BackendUnavailableError(msg)


def _import_lmstudio() -> Any:
    try:
        import lmstudio as lms
    except ImportError as exc:
        msg = "lmstudio package is required for the LM Studio backend extra"
        raise BackendUnavailableError(msg) from exc
    _silence_lmstudio_sdk_logs()
    return lms


def _silence_lmstudio_sdk_logs() -> None:
    """Drop LM Studio SDK websocket chatter (class-named loggers at INFO)."""
    # SDK uses type(self).__name__ as logger names (not under ``lmstudio.*``).
    for name in (
        "_AsyncLMStudioWebsocket",
        "AsyncWebsocketHandler",
        "AsyncWebsocketThread",
        "_SyncLMStudioWebsocket",
        "SyncWebsocketHandler",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
