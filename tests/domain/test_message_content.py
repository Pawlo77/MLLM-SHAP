"""Tests for multimodal message content parts.

@see [[knowledge/architecture/domain]]
"""

import hashlib
from pathlib import Path

import pytest

from mllm_shap.backends._chat import snapshot_has_audio
from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    ImagePart,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role


def _snapshot(*messages: Message) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((Turn(messages=messages),))


def test_text_keyword_builds_a_single_text_part() -> None:
    message = Message(role=Role.USER, text="hello world")
    assert message.content == (TextPart("hello world"),)
    assert message.text == "hello world"


def test_text_joins_text_parts_and_skips_audio() -> None:
    message = Message(
        role=Role.USER,
        content=(TextPart("listen"), AudioPart(b"\x00\x01", 16_000), TextPart("now")),
    )
    assert message.text == "listen now"


def test_message_requires_exactly_one_of_text_or_content() -> None:
    with pytest.raises(ValueError, match="text or content"):
        Message(role=Role.USER)
    with pytest.raises(ValueError, match="text or content"):
        Message(role=Role.USER, text="a", content=(TextPart("b"),))


def test_with_text_keeps_audio_parts() -> None:
    audio = AudioPart(b"\x00", 8_000)
    message = Message(role=Role.USER, content=(TextPart("a b"), audio))
    assert message.with_text("a").content == (TextPart("a"), audio)


def test_text_only_snapshot_ids_are_unchanged() -> None:
    """Snapshot ids of text-only conversations match the pre-content-part digest."""
    snapshot = _snapshot(Message(role=Role.USER, text="Who are you?"))
    payload = '[[{"role":"user","text":"Who are you?"}]]'
    assert snapshot.snapshot_id == hashlib.sha256(payload.encode()).hexdigest()[:16]


def test_digest_hashes_audio_bytes(tmp_path: Path) -> None:
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"\x01\x02")
    from_path = _snapshot(
        Message(role=Role.USER, content=(TextPart("x"), AudioPart(clip, 16_000)))
    )
    from_bytes = _snapshot(
        Message(role=Role.USER, content=(TextPart("x"), AudioPart(b"\x01\x02", 16_000)))
    )
    other = _snapshot(
        Message(role=Role.USER, content=(TextPart("x"), AudioPart(b"\x01\x03", 16_000)))
    )
    text_only = _snapshot(Message(role=Role.USER, text="x"))
    assert from_path.snapshot_id == from_bytes.snapshot_id
    assert from_bytes.snapshot_id != other.snapshot_id
    assert from_bytes.snapshot_id != text_only.snapshot_id


def test_snapshot_has_audio_checks_part_types() -> None:
    assert not snapshot_has_audio(_snapshot(Message(role=Role.USER, text="audio:x")))
    assert snapshot_has_audio(
        _snapshot(Message(role=Role.USER, content=(AudioPart(b"", 16_000),)))
    )


def test_with_media_replaces_the_indexed_part_of_its_modality() -> None:
    first, second, image = AudioPart(b"a", 8000), AudioPart(b"b", 8000), ImagePart(b"i")
    message = Message(
        role=Role.USER, content=(TextPart("hi"), first, image, second), speaker="Ann"
    )
    new, picture = AudioPart(b"n", 8000), ImagePart(b"p")
    assert message.with_media(new).content == (TextPart("hi"), new, image, second)
    assert message.with_media(new, 1).content == (TextPart("hi"), first, image, new)
    assert message.with_media(picture).content == (
        TextPart("hi"),
        first,
        picture,
        second,
    )
    assert message.with_media(new).speaker == "Ann"


def test_with_media_refuses_a_missing_part() -> None:
    with pytest.raises(ValueError, match="no audio part 0"):
        Message(role=Role.USER, text="hi").with_media(AudioPart(b"n", 8000))
    with pytest.raises(ValueError, match="no image part 1"):
        Message(role=Role.USER, content=(ImagePart(b"i"),)).with_media(
            ImagePart(b"j"), 1
        )


def test_with_text_parts_keeps_each_text_part_in_place() -> None:
    image = ImagePart(b"i")
    message = Message(role=Role.USER, content=(TextPart("a b"), image, TextPart("c d")))
    assert message.with_text_parts(("a", "d")).content == (
        TextPart("a"),
        image,
        TextPart("d"),
    )
    with pytest.raises(ValueError, match="2 text parts"):
        message.with_text_parts(("a",))


def test_text_skips_empty_text_parts() -> None:
    message = Message(
        role=Role.USER, content=(TextPart("a"), TextPart(""), TextPart("c"))
    )
    assert message.text == "a c"
