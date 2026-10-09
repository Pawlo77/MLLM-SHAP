"""Direct tests for whitespace token spans and span rendering.

@see [[src/nlp_shap/masking/tokens.py]]
"""

import pytest

from nlp_shap.domain.conversation import ConversationSnapshot, Message, Turn
from nlp_shap.domain.enums import Role
from nlp_shap.domain.players import PlayerSpan
from nlp_shap.masking.tokens import render_spans, token_spans


def _snapshot(*texts: str) -> ConversationSnapshot:
    return ConversationSnapshot.from_turns(
        tuple(Turn(messages=(Message(role=Role.USER, text=text),)) for text in texts)
    )


def _render(text: str, replacements: list[str | None]) -> str:
    snapshot = _snapshot(text)
    spans = [span for _, span in token_spans(snapshot)]
    return render_spans(snapshot, spans, replacements).messages()[0].text


def test_token_spans_cover_every_token_across_turns() -> None:
    """Spans carry flattened message indices, offsets and turn/message ids."""
    snapshot = _snapshot("  hi there", "bye")
    spans = token_spans(snapshot)

    assert [player_id.split(":", 1)[1] for player_id, _ in spans] == [
        "0:0:0",
        "0:0:1",
        "1:0:0",
    ]
    assert [span for _, span in spans] == [
        PlayerSpan(message_index=0, start=2, end=4, text="hi"),
        PlayerSpan(message_index=0, start=5, end=10, text="there"),
        PlayerSpan(message_index=1, start=0, end=3, text="bye"),
    ]


@pytest.mark.parametrize(
    ("replacements", "expected"),
    [
        ([None, "b", "c"], "b c"),
        (["a", None, "c"], "a c"),
        (["a", "b", None], "a b"),
        ([None, None, None], ""),
        (["_", "b", "_"], "_ b _"),
    ],
)
def test_render_spans_merges_whitespace_around_deletions(
    replacements: list[str | None], expected: str
) -> None:
    assert _render("a b c", replacements) == expected


def test_render_spans_keeps_uncovered_text_and_original_whitespace() -> None:
    """Only the first token is a player; the rest of the message stays verbatim."""
    snapshot = _snapshot("a  b,\tc")
    first = token_spans(snapshot)[0][1]

    rendered = render_spans(snapshot, [first], ["Z"])

    assert rendered.messages()[0].text == "Z  b,\tc"


def test_render_spans_leaves_untouched_messages_identical() -> None:
    snapshot = _snapshot("a b", "c d")
    first_message_spans = [
        span for _, span in token_spans(snapshot) if span.message_index == 0
    ]

    rendered = render_spans(snapshot, first_message_spans, [None, "B"])

    assert [message.text for message in rendered.messages()] == ["B", "c d"]
    assert rendered.messages()[1] is snapshot.messages()[1]


def test_render_spans_rejects_mismatched_input() -> None:
    snapshot = _snapshot("a b")
    spans = [span for _, span in token_spans(snapshot)]

    with pytest.raises(ValueError, match="replacement count"):
        render_spans(snapshot, spans, ["x"])
    stale = PlayerSpan(message_index=0, start=0, end=1, text="z")
    with pytest.raises(ValueError, match="do not match"):
        render_spans(snapshot, [stale], ["x"])
