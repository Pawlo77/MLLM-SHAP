"""Conversation view: messages under role headers, players coloured in place.

@see [[src/nlp_shap/viz/conversation.py]]
"""

import dataclasses

import pytest

pytest.importorskip("matplotlib")

from nlp_shap import (
    ConversationSnapshot,
    ExplainConfig,
    ExplainRunner,
    ExplainRunOutput,
    Message,
    Role,
    Turn,
)
from nlp_shap.viz import plot_conversation, render_conversation_html

THREAD = ConversationSnapshot.from_turns((
    Turn(messages=(Message(role=Role.SYSTEM, text="Answer briefly."),)),
    Turn(messages=(Message(role=Role.USER, speaker="Alice", text="Laptop broken."),)),
    Turn(messages=(Message(role=Role.ASSISTANT, text="Sorry to hear."),)),
    Turn(messages=(Message(role=Role.USER, speaker="Bob", text="Refund?"),)),
))


def _explain(players: str, **options: object) -> ExplainRunOutput:
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "players": players,
            "plugin_options": {players: options} if options else {},
            "archive": {"path": ""},
        },
    })
    output = ExplainRunner(config).explain_sync(THREAD)
    signs = tuple((-1.0) ** i * (i + 1) / 10 for i in range(len(output.result.values)))
    return dataclasses.replace(
        output, result=dataclasses.replace(output.result, values=signs)
    )


def test_html_shows_every_message_with_only_players_coloured() -> None:
    output = _explain("tokens", roles=["user"])
    assert output.player_set is not None
    html = render_conversation_html(output, THREAD, output.player_set)
    for text in ("Answer briefly.", "Sorry to hear.", "Laptop", "broken.", "Refund?"):
        assert text in html
    assert "user/Alice 2" in html
    assert "assistant 3" in html
    assert html.count("nlp-shap-player") == len(output.player_set.spans) == 3
    assert "+0.100" in html
    assert "-0.200" in html


def test_figure_draws_headers_players_and_context() -> None:
    output = _explain("tokens", roles=["user"])
    assert output.player_set is not None
    figure = plot_conversation(output, THREAD, output.player_set)
    texts = [t for t in figure.axes[0].texts if t.get_text()]
    shown = [t.get_text() for t in texts]
    assert "system 1" in shown
    assert "user/Bob 4" in shown
    assert "Answer" in shown
    assert "Refund?" in shown
    boxed = [t for t in texts if t.get_bbox_patch() is not None]
    assert sorted(t.get_text() for t in boxed) == ["Laptop", "Refund?", "broken."]


def test_message_players_colour_whole_messages() -> None:
    output = _explain("messages")
    assert output.player_set is not None
    html = render_conversation_html(output, THREAD, output.player_set)
    assert html.count("nlp-shap-player") == 4


def test_values_must_match_the_players() -> None:
    output = _explain("messages")
    assert output.player_set is not None
    short = dataclasses.replace(
        output, result=dataclasses.replace(output.result, values=(0.1,))
    )
    with pytest.raises(ValueError, match="same length"):
        render_conversation_html(short, THREAD, output.player_set)


def test_audio_messages_get_a_waveform_strip_on_the_same_colour_scale() -> None:
    """Typed and spoken players share one scale; the spoken ones also on the audio."""
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    from nlp_shap import AudioPart, PluginGroup, TextPart
    from nlp_shap.alignment.partitions import SgpaSegmentPartitioner
    from nlp_shap.alignment.sgpa import SpectrogramGuidedAligner
    from nlp_shap.plugins import default_registry
    from nlp_shap.plugins.context import PluginContext
    from nlp_shap.viz.colors import token_background_rgba
    from tests.alignment._fakes import RATE, FakeCtc, speech

    snapshot = ConversationSnapshot.from_turns((
        Turn(
            messages=(
                Message(role=Role.USER, text="What was said?"),
                Message(
                    role=Role.USER,
                    content=(TextPart("ab — c"), AudioPart(speech(), RATE)),
                ),
            )
        ),
    ))
    config = ExplainConfig.model_validate({
        "backend": {"kind": "mock", "model_id": "stub"},
        "explanation": {
            "players": "sgpa_segments",
            "plugin_options": {"sgpa_segments": {"text_roles": ["user"]}},
            "archive": {"path": ""},
        },
    })
    registry = default_registry()
    registry.register(
        PluginGroup.PARTITIONS,
        "sgpa_segments",
        lambda context: SgpaSegmentPartitioner(
            PluginContext(config=config, options={"text_roles": ["user"]}),
            aligner=SpectrogramGuidedAligner(FakeCtc()),
        ),
        replace=True,
    )
    output = ExplainRunner(config, registry=registry).explain_sync(snapshot)
    values = (0.1, -0.2, 0.3, 0.5, -0.4)
    output = dataclasses.replace(
        output, result=dataclasses.replace(output.result, values=values)
    )
    assert output.player_set is not None
    figure = plot_conversation(output, snapshot, output.player_set)
    text_axis, audio_axis = figure.axes
    assert audio_axis.get_title(loc="left") == "user 2 (audio)"
    segments = [p for p in audio_axis.patches if p.get_label() == "segment"]
    assert [p.get_facecolor() for p in segments] == pytest.approx([
        token_background_rgba(v, 0.5) for v in values[3:]
    ])
    boxed = {t.get_text() for t in text_axis.texts if t.get_bbox_patch() is not None}
    assert boxed == {"What", "was", "said?", "ab", "c"}
