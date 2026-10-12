"""Tests for conversation snapshots and their JSON form.

@see [[src/mllm_shap/domain/conversation.py]]
"""

import base64
import json
from pathlib import Path

import pytest

from mllm_shap.domain.conversation import (
    AudioPart,
    ConversationSnapshot,
    Message,
    TextPart,
    Turn,
)
from mllm_shap.domain.enums import Role


def test_conversation_snapshot_from_turns_is_stable() -> None:
    """Identical turns produce the same snapshot identifier."""
    turn = Turn(
        messages=(
            Message(role=Role.USER, text="Who"),
            Message(role=Role.USER, text="are"),
            Message(role=Role.USER, text="you?"),
        )
    )
    first = ConversationSnapshot.from_turns((turn,))
    second = ConversationSnapshot.from_turns((turn,))
    assert first.snapshot_id == second.snapshot_id
    assert first == second


def test_conversation_snapshot_rejects_empty_turns() -> None:
    """Snapshots require at least one turn."""
    with pytest.raises(ValueError, match="at least one turn"):
        ConversationSnapshot(turns=(), snapshot_id="abc")


def test_turn_rejects_empty_messages() -> None:
    """Turns require at least one message."""
    with pytest.raises(ValueError, match="at least one message"):
        Turn(messages=())


def test_snapshot_json_round_trip() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.SYSTEM, text="Be brief."),
                Message(role=Role.USER, text="Hi"),
            )
        ),
    ))
    restored = ConversationSnapshot.from_json(snapshot.to_json())
    assert restored == snapshot
    assert json.loads(snapshot.to_json()) == {
        "turns": [
            {
                "messages": [
                    {"role": "system", "text": "Be brief."},
                    {"role": "user", "text": "Hi"},
                ]
            }
        ]
    }


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("not json", ValueError),
        ('{"messages": []}', TypeError),
        ('{"turns": [{"messages": [{"role": "robot", "text": "x"}]}]}', ValueError),
        ('{"turns": []}', ValueError),
    ],
)
def test_snapshot_from_json_rejects_bad_input(text: str, error: type) -> None:
    with pytest.raises(error):
        ConversationSnapshot.from_json(text)


def test_snapshot_json_round_trips_custom_snapshot_id() -> None:
    turns = (Turn(messages=(Message(role=Role.USER, text="Hi"),)),)
    snapshot = ConversationSnapshot(turns=turns, snapshot_id="ticket-42")

    restored = ConversationSnapshot.from_json(snapshot.to_json())

    assert restored.snapshot_id == "ticket-42"
    assert restored == snapshot
    assert json.loads(snapshot.to_json())["snapshot_id"] == "ticket-42"


def test_snapshot_json_omits_content_derived_id() -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="Hi"),)),
    ))
    assert "snapshot_id" not in json.loads(snapshot.to_json())
    assert ConversationSnapshot.from_json(snapshot.to_json()) == snapshot


def _parts_snapshot(*content: TextPart | AudioPart) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, content=content, speaker="ann"),)),
    ))


def test_snapshot_json_round_trips_text_and_audio_parts(tmp_path: Path) -> None:
    """Multi-part messages use ``"content"``; audio goes by path or base64 data."""
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF-path")
    snapshot = _parts_snapshot(
        TextPart("say it"),
        AudioPart(clip, 16_000),
        AudioPart(b"RIFF-bytes", 8_000),
    )
    payload = json.loads(snapshot.to_json())
    assert payload["turns"][0]["messages"][0] == {
        "role": "user",
        "speaker": "ann",
        "content": [
            {"type": "text", "text": "say it"},
            {"type": "audio", "path": str(clip), "sample_rate": 16_000},
            {
                "type": "audio",
                "data": base64.b64encode(b"RIFF-bytes").decode("ascii"),
                "sample_rate": 8_000,
            },
        ],
    }
    assert ConversationSnapshot.from_json(snapshot.to_json()) == snapshot


def test_snapshot_json_resolves_audio_paths_against_base_dir(tmp_path: Path) -> None:
    (tmp_path / "clip.wav").write_bytes(b"RIFF")
    text = json.dumps({
        "turns": [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "path": "clip.wav", "sample_rate": 16000}
                        ],
                    }
                ]
            }
        ]
    })
    snapshot = ConversationSnapshot.from_json(text, base_dir=tmp_path)
    (part,) = snapshot.messages()[0].content
    assert part == AudioPart(tmp_path / "clip.wav", 16_000)
    assert snapshot == _audio_only(tmp_path / "clip.wav")


def _audio_only(path: Path) -> ConversationSnapshot:
    message = Message(role=Role.USER, content=(AudioPart(path, 16_000),))
    return ConversationSnapshot.from_turns((Turn(messages=(message,)),))


@pytest.mark.parametrize(
    ("message", "error", "match"),
    [
        ({"role": "user", "text": "a", "content": []}, TypeError, "exactly one"),
        ({"role": "user", "content": "a"}, TypeError, "list"),
        ({"role": "user", "text": 5}, TypeError, '"text" must be a string'),
        ({"role": "user", "content": [{"type": "video"}]}, ValueError, "video"),
        (
            {"role": "user", "content": [{"type": "audio", "path": "x.wav"}]},
            TypeError,
            "sample_rate",
        ),
        (
            {"role": "user", "content": [{"type": "audio", "sample_rate": 1}]},
            TypeError,
            "path",
        ),
        ({"role": "user", "content": [{"type": "text"}]}, TypeError, "text"),
        (
            {
                "role": "user",
                "content": [{"type": "audio", "data": "%%", "sample_rate": 1}],
            },
            ValueError,
            "base64",
        ),
    ],
)
def test_snapshot_json_rejects_bad_content_parts(
    message: dict[str, object], error: type, match: str
) -> None:
    text = json.dumps({"turns": [{"messages": [message]}]})
    with pytest.raises(error, match=match):
        ConversationSnapshot.from_json(text)


def test_snapshot_from_json_rejects_non_string_snapshot_id() -> None:
    with pytest.raises(TypeError, match="snapshot_id"):
        ConversationSnapshot.from_json(
            '{"snapshot_id": 3, "turns": [{"messages": '
            '[{"role": "user", "text": "x"}]}]}'
        )


def _speaker_snapshot(speaker: str | None) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, text="Hi", speaker=speaker),)),
    ))


def test_snapshot_json_round_trips_speaker() -> None:
    snapshot = _speaker_snapshot("alice")

    restored = ConversationSnapshot.from_json(snapshot.to_json())

    assert restored == snapshot
    assert restored.messages()[0].speaker == "alice"
    message = json.loads(snapshot.to_json())["turns"][0]["messages"][0]
    assert message == {"role": "user", "text": "Hi", "speaker": "alice"}
    assert "speaker" not in _speaker_snapshot(None).to_json()


def test_snapshot_id_changes_with_speaker() -> None:
    ids = {_speaker_snapshot(s).snapshot_id for s in (None, "alice", "bob")}
    assert len(ids) == 3


def test_with_text_keeps_speaker() -> None:
    message = Message(role=Role.USER, text="Hi", speaker="alice")
    assert message.with_text("Bye").speaker == "alice"


@pytest.mark.parametrize("speaker", ["", "two\nlines"])
def test_message_rejects_bad_speaker(speaker: str) -> None:
    with pytest.raises(ValueError, match="speaker"):
        Message(role=Role.USER, text="Hi", speaker=speaker)


def test_snapshot_from_json_rejects_non_string_speaker() -> None:
    with pytest.raises(TypeError, match="speaker"):
        ConversationSnapshot.from_json(
            '{"turns": [{"messages": [{"role": "user", "text": "x", "speaker": 1}]}]}'
        )
