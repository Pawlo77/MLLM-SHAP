"""Model inputs for audio language models through a Hugging Face processor.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_transformers_audio.py]]
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...alignment.io import decode_audio
from ...domain.conversation import AudioPart, ConversationSnapshot, Message
from .._chat import chat_role, message_chat_text
from .chat import render_prompt


def processor_inputs(processor: Any, snapshot: ConversationSnapshot) -> Any:
    """Encode ``snapshot`` with its audio for ``processor``.

    A processor whose chat template mentions audio gets structured audio and text
    parts; otherwise each audio part becomes ``processor.audio_token`` in the
    rendered prompt and the waveforms go to ``processor(text=..., audio=...)``.
    Audio parts come before the message text, and an audio part left empty by
    masking is dropped.
    """
    rate = _sampling_rate(processor)
    audio = {
        id(message): [w for w in _waveforms(message, rate) if w.size]
        for message in snapshot.messages()
    }
    if _template_has_audio(processor):
        conversation = [
            {"role": chat_role(m.role), "content": _parts(m, audio[id(m)])}
            for m in snapshot.messages()
        ]
        return processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    messages = [
        {
            "role": chat_role(m.role),
            "content": processor.audio_token * len(audio[id(m)]) + message_chat_text(m),
        }
        for m in snapshot.messages()
    ]
    waveforms = [w for m in snapshot.messages() for w in audio[id(m)]]
    return processor(
        text=render_prompt(processor.tokenizer, messages),
        audio=waveforms or None,
        return_tensors="pt",
    )


def _parts(message: Message, waveforms: list[NDArray[np.float32]]) -> list[Any]:
    parts: list[Any] = [{"type": "audio", "audio": w} for w in waveforms]
    text = message_chat_text(message)
    if text:
        parts.append({"type": "text", "text": text})
    return parts


def _waveforms(message: Message, rate: int) -> list[NDArray[np.float32]]:
    return [
        decode_audio(part.data(), rate)[0]
        for part in message.content
        if isinstance(part, AudioPart)
    ]


def _sampling_rate(processor: Any) -> int:
    # Processors name their audio component differently (Granite: audio_processor).
    extractor: Any = getattr(processor, "feature_extractor", None) or getattr(
        processor, "audio_processor", None
    )
    return int(extractor.sampling_rate)


def _template_has_audio(processor: Any) -> bool:
    templates = (
        getattr(processor, "chat_template", None),
        getattr(processor.tokenizer, "chat_template", None),
    )
    return any(isinstance(t, str) and "audio" in t for t in templates)
