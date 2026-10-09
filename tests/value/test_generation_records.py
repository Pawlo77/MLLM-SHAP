"""Tests for building generation records from generated text.

@see [[src/nlp_shap/value/generation_records.py]]
"""

import pytest

from nlp_shap.value.generation_records import (
    EMBED_DIM,
    generation_record_from_text,
    hash_embedding,
    stable_token_id,
)


def test_record_token_rows_come_from_generated_text() -> None:
    record = generation_record_from_text("alpha beta alpha")
    assert record.text_token_rows == (
        (stable_token_id("alpha"),),
        (stable_token_id("beta"),),
        (stable_token_id("alpha"),),
    )
    assert len(record.embedding) == EMBED_DIM


def test_stable_token_id_separates_shared_prefixes() -> None:
    assert stable_token_id("alpha") != stable_token_id("alphabet")


def test_contextual_embedding_is_a_separate_hash_of_the_text() -> None:
    """C15: the contextual vector depends on the text alone, not on a snapshot id."""
    record = generation_record_from_text("alpha beta")
    assert record.embedding == hash_embedding("alpha beta")
    assert record.contextual_embedding == hash_embedding("alpha beta\0")
    assert record.contextual_embedding != record.embedding


def test_context_id_is_deprecated_and_ignored() -> None:
    with pytest.deprecated_call(match="context_id"):
        salted = generation_record_from_text("alpha beta", context_id="snap-1")
    assert salted == generation_record_from_text("alpha beta")


def test_hash_embedding_is_deterministic_and_bounded() -> None:
    vector = hash_embedding("alpha")
    assert vector == hash_embedding("alpha")
    assert vector != hash_embedding("beta")
    assert all(-1.0 <= value <= 1.0 for value in vector)
    assert len(vector) == EMBED_DIM


def test_empty_text_has_no_token_rows() -> None:
    assert generation_record_from_text("").text_token_rows == ()
