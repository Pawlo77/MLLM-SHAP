"""Chat-completions request and response helpers.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_api.py]]
"""

import base64
import hashlib
import json
from typing import Any

from ...alignment.io import decode_audio, encode_wav
from ...domain.config import GenerationConfig
from ...domain.conversation import AudioPart, ConversationSnapshot
from ...errors import BackendUnavailableError
from .._chat import chat_role, message_chat_text


def snapshot_to_api_messages(snapshot: ConversationSnapshot) -> list[dict[str, Any]]:
    """Convert ``snapshot`` to chat messages; audio becomes ``input_audio`` parts.

    A message with audio sends its text part first, then one base64 WAV part per
    ``AudioPart``; non-WAV audio is re-encoded (``audio`` extra).
    """
    messages: list[dict[str, Any]] = []
    for message in snapshot.messages():
        text = message_chat_text(message)
        audio = [part for part in message.content if isinstance(part, AudioPart)]
        content: str | list[dict[str, Any]] = text
        if audio:
            content = [{"type": "text", "text": text}] if text else []
            content += [_input_audio(part) for part in audio]
        messages.append({"role": chat_role(message.role), "content": content})
    return messages


def _input_audio(part: AudioPart) -> dict[str, Any]:
    data = part.data()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        data = encode_wav(*decode_audio(data))
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


def build_chat_payload(
    model_id: str,
    messages: list[dict[str, Any]],
    generation: GenerationConfig,
) -> dict[str, Any]:
    """Build an OpenAI-style chat-completions JSON body that asks for logprobs."""
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": generation.max_new_tokens,
        "logprobs": True,
    }
    if generation.temperature > 0.0:
        payload["temperature"] = float(generation.temperature)
        if generation.top_k > 0:
            payload["top_p"] = 0.95
    else:
        payload["temperature"] = 0.0
    return payload


def payload_cache_key(payload: dict[str, Any]) -> str:
    """Return a stable SHA256 key for deterministic request deduplication."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_message_content(data: dict[str, Any]) -> str:
    """Parse assistant text from a chat-completions JSON response."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = "unexpected chat/completions response payload"
        raise BackendUnavailableError(msg) from exc
    if not isinstance(content, str):
        msg = "chat/completions content was not a string"
        raise BackendUnavailableError(msg)
    return content.strip()


def extract_logprobs(data: dict[str, Any]) -> tuple[float, ...] | None:
    """Return ``choices[0].logprobs.content[*].logprob``, or ``None`` when absent.

    Servers that ignore the ``logprobs`` request field (or send another shape)
    give ``None``; the ``logprob`` value function then refuses the record.
    """
    try:
        content = data["choices"][0]["logprobs"]["content"]
        values = tuple(float(item["logprob"]) for item in content)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return values
