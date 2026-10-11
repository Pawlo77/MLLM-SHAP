"""Unit tests for LM Studio model resolution and download helpers.

@see [[src/mllm_shap/backends/lmstudio/models.py]]
"""

from types import SimpleNamespace

import pytest

from mllm_shap.backends.lmstudio.models import (
    match_downloaded_model_key,
    pick_download_option,
    pick_search_result,
)
from mllm_shap.errors import BackendUnavailableError


def test_match_downloaded_model_key_finds_normalized_match() -> None:
    """Downloaded model keys match configured ids with hyphen/underscore variants."""
    downloaded = (
        SimpleNamespace(model_key="qwen2.5-3b-instruct"),
        SimpleNamespace(model_key="gemma-2-2b-it"),
    )
    assert (
        match_downloaded_model_key(downloaded, "qwen2_5_3b_instruct")
        == "qwen2.5-3b-instruct"
    )


def test_match_downloaded_model_key_returns_none_when_missing() -> None:
    """Unknown model ids do not match unrelated downloads."""
    downloaded = (SimpleNamespace(model_key="llama-3.2-3b-instruct"),)
    assert match_downloaded_model_key(downloaded, "qwen2.5-3b-instruct") is None


def test_pick_search_result_prefers_exact_name_match() -> None:
    """Repository search picks the entry whose name best matches the target id."""
    results = (
        SimpleNamespace(
            search_result=SimpleNamespace(
                name="Some other model",
                exact=False,
            )
        ),
        SimpleNamespace(
            search_result=SimpleNamespace(
                name="Qwen2.5 3B Instruct",
                exact=True,
            )
        ),
    )
    picked = pick_search_result(results, "qwen2.5-3b-instruct")
    assert picked.search_result.name == "Qwen2.5 3B Instruct"


def test_pick_search_result_refuses_hits_that_do_not_match() -> None:
    """With no exact flag and no name match, nothing is picked for download."""
    results = (
        SimpleNamespace(search_result=SimpleNamespace(name="Llama 3", exact=False)),
        SimpleNamespace(search_result=SimpleNamespace(name="Gemma 2", exact=False)),
    )
    with pytest.raises(BackendUnavailableError, match="Llama 3, Gemma 2"):
        pick_search_result(results, "qwen2.5-3b-instruct")


def test_pick_download_option_prefers_requested_quantization() -> None:
    """Download options select the configured GGUF quantization when present."""
    options = (
        SimpleNamespace(info=SimpleNamespace(quantization="Q8_0", name="q8")),
        SimpleNamespace(info=SimpleNamespace(quantization="Q4_K_M", name="q4")),
    )
    picked = pick_download_option(options, "Q4_K_M")
    assert picked.info.quantization == "Q4_K_M"


def test_pick_download_option_falls_back_to_recommended() -> None:
    """When quantization is absent, a recommended option is used."""
    options = (
        SimpleNamespace(
            info=SimpleNamespace(quantization=None, name="fp16", recommended=False),
        ),
        SimpleNamespace(
            info=SimpleNamespace(quantization=None, name="q4", recommended=True),
        ),
    )
    picked = pick_download_option(options, "Q4_K_M")
    assert picked.info.name == "q4"


def test_pick_download_option_raises_when_empty() -> None:
    """Empty download option lists surface a clear backend error."""
    with pytest.raises(BackendUnavailableError, match="no download options"):
        pick_download_option((), "Q4_K_M")
