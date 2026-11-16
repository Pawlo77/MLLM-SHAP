"""Infill absence policy with a stand-in filler.

@see [[src/mllm_shap/masking/infill.py]]
"""

import pytest

from mllm_shap.domain.coalition import CoalitionMask
from mllm_shap.domain.conversation import ConversationSnapshot
from mllm_shap.errors import BackendUnavailableError
from mllm_shap.masking.infill import HfMaskedLmFiller, InfillPolicy
from mllm_shap.masking.partitions import TokenPartitioner


class FakeFiller:
    """Fills every mask with ``fill<n>`` where ``n`` counts calls to ``fill``."""

    mask_token = "<mask>"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def fill(self, text: str) -> list[str]:
        self.calls.append(text)
        return [f"fill{i}" for i in range(text.count(self.mask_token))]


def test_absent_words_are_replaced_by_the_fillers_words(
    sample_snapshot: ConversationSnapshot,
) -> None:
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    filler = FakeFiller()
    rendered = InfillPolicy(filler=filler).apply(sample_snapshot, players, mask)
    assert rendered.turns[0].messages[0].text == "Who fill0 you?"
    assert rendered.turns[0].messages[1].text == "I am helpful."
    assert filler.calls == ["Who <mask> you?"]


def test_several_absent_words_are_filled_in_one_call_in_order(
    sample_snapshot: ConversationSnapshot,
) -> None:
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((False, False, True, True, True, True))
    filler = FakeFiller()
    rendered = InfillPolicy(filler=filler).apply(sample_snapshot, players, mask)
    assert rendered.turns[0].messages[0].text == "fill0 fill1 you?"
    assert len(filler.calls) == 1


def test_the_same_coalition_renders_the_same_text_and_is_cached(
    sample_snapshot: ConversationSnapshot,
) -> None:
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((True, False, True, True, True, True))
    filler = FakeFiller()
    policy = InfillPolicy(filler=filler)
    first = policy.apply(sample_snapshot, players, mask)
    second = policy.apply(sample_snapshot, players, mask)
    assert first == second
    assert len(filler.calls) == 1


def test_the_empty_coalition_has_no_mask_token_left(
    sample_snapshot: ConversationSnapshot,
) -> None:
    players = TokenPartitioner().partition(sample_snapshot)
    mask = CoalitionMask.from_sequence((False,) * players.num_players)
    rendered = InfillPolicy(filler=FakeFiller()).apply(sample_snapshot, players, mask)
    assert all("<mask>" not in m.text for t in rendered.turns for m in t.messages)


def test_a_missing_transformers_install_names_the_extra() -> None:
    filler = HfMaskedLmFiller("model-that-does-not-exist-locally")
    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(BackendUnavailableError, match=r"mllm-shap\[infill\]"):
            _ = filler.mask_token


@pytest.mark.gpu
def test_a_real_masked_model_fills_deterministically() -> None:
    """Needs the infill extra and downloads a tiny public checkpoint."""
    pytest.importorskip("transformers")
    filler = HfMaskedLmFiller("hf-internal-testing/tiny-random-RobertaForMaskedLM")
    text = f"The capital of {filler.mask_token} is {filler.mask_token}."
    first = filler.fill(text)
    assert len(first) == 2
    assert all(word for word in first)
    assert filler.fill(text) == first
