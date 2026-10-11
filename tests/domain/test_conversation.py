"""Tests for conversation snapshots and their JSON form.

@see [[src/mllm_shap/domain/conversation.py]]
"""

import json

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


@pytest.mark.parametrize(
    "content",
    [
        (TextPart("one"), TextPart("two")),
        (TextPart("hi"), AudioPart(b"\x00", 16_000)),
    ],
)
def test_snapshot_to_json_rejects_multi_part_and_audio_messages(
    content: tuple[TextPart | AudioPart, ...],
) -> None:
    snapshot = ConversationSnapshot.from_turns((
        Turn(messages=(Message(role=Role.USER, content=content),)),
    ))
    with pytest.raises(ValueError, match="single text part"):
        snapshot.to_json()


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
