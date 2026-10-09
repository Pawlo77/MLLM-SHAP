"""Resolve and download LM Studio models through the ``lmstudio`` SDK.

@domain backends
@see [[knowledge/architecture/backends]]
@tested-by [[tests/backends/test_lmstudio_models.py]]
"""

import logging
import re
from collections.abc import Callable, Sequence
from typing import Any

from ...errors import BackendUnavailableError
from .options import LmStudioOptions

logger = logging.getLogger(__name__)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def normalize_model_token(value: str) -> str:
    """Normalize a model identifier for fuzzy comparison."""
    return _TOKEN_SPLIT_RE.sub("", value.casefold())


def match_downloaded_model_key(
    downloaded: Sequence[Any],
    model_id: str,
) -> str | None:
    """Return the downloaded ``model_key`` for ``model_id``, if any.

    Candidates are tried in three stages, each on normalized names: an exact
    match, then a key whose last ``/`` segment starts with ``model_id``, then a
    key containing ``model_id``. The first stage with a single match wins.

    Raises:
        BackendUnavailableError: If a stage matches more than one key.
    """
    target = normalize_model_token(model_id)
    if not target:
        return None
    keys = [str(getattr(entry, "model_key", "")) for entry in downloaded]
    stages: tuple[Callable[[str], bool], ...] = (
        lambda key: normalize_model_token(key) == target,
        lambda key: normalize_model_token(key.split("/")[-1]).startswith(target),
        lambda key: target in normalize_model_token(key),
    )
    for matches in stages:
        candidates = [key for key in keys if key and matches(key)]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            msg = (
                f"LM Studio model id {model_id!r} is ambiguous; downloaded "
                f"candidates: {', '.join(sorted(candidates))}. Set backend.model_id "
                "to one of them."
            )
            raise BackendUnavailableError(msg)
    return None


def pick_search_result(results: Sequence[Any], model_id: str) -> Any:
    """Choose the best repository search hit for ``model_id``.

    Raises:
        BackendUnavailableError: If there are no hits or none is flagged exact or
            has a name matching ``model_id``.
    """
    if not results:
        msg = f"LM Studio repository returned no results for {model_id!r}"
        raise BackendUnavailableError(msg)
    target = normalize_model_token(model_id)

    def score(entry: Any) -> tuple[int, int]:
        data = entry.search_result
        name = normalize_model_token(str(getattr(data, "name", "")))
        exact = 1 if bool(getattr(data, "exact", False)) else 0
        name_match = 2 if name == target else 1 if target and target in name else 0
        return (exact, name_match)

    best = max(results, key=score)
    if score(best) == (0, 0):
        names = ", ".join(str(getattr(r.search_result, "name", "")) for r in results)
        msg = (
            f"LM Studio repository has no model matching {model_id!r} "
            f"(hits: {names}); set backend.model_id to one of them"
        )
        raise BackendUnavailableError(msg)
    return best


def pick_download_option(options: Sequence[Any], quantization: str) -> Any:
    """Choose a repository download option for the requested quantization."""
    if not options:
        msg = "LM Studio model has no download options"
        raise BackendUnavailableError(msg)
    target = quantization.casefold()
    for option in options:
        value = getattr(option.info, "quantization", None)
        if isinstance(value, str) and value.casefold() == target:
            return option
    for option in options:
        if bool(getattr(option.info, "recommended", False)):
            return option
    for option in options:
        name = str(getattr(option.info, "name", "")).casefold()
        if target in name:
            return option
    return options[0]


async def resolve_model_key(
    client: Any,
    model_id: str,
    options: LmStudioOptions,
) -> str:
    """Return a loadable LM Studio model key, downloading when configured."""
    downloaded = await client.llm.list_downloaded()
    matched = match_downloaded_model_key(downloaded, model_id)
    if matched is not None:
        logger.info(
            "Using downloaded LM Studio model %r for configured id %r",
            matched,
            model_id,
        )
        return matched

    try:
        await client.llm.get_model_info(model_id)
    except Exception:
        if not options.auto_download:
            msg = (
                f"LM Studio model {model_id!r} is not downloaded "
                "and auto_download is disabled"
            )
            raise BackendUnavailableError(msg) from None
        return await _download_model(client, model_id, options)

    return model_id


async def _download_model(
    client: Any,
    model_id: str,
    options: LmStudioOptions,
) -> str:
    search_term = options.hub_search or model_id
    logger.info(
        "Downloading LM Studio model for %r (search=%r, quant=%s)",
        model_id,
        search_term,
        options.quantization,
    )
    results = await client.repository.search_models(
        search_term=search_term,
        limit=10,
        compatibility_types=["gguf"],
    )
    chosen = pick_search_result(results, model_id)
    download_options = await chosen.get_download_options()
    option = pick_download_option(download_options, options.quantization)
    quant = getattr(option.info, "quantization", None) or options.quantization
    model_name = getattr(chosen.search_result, "name", search_term)
    logger.info("Downloading %s (%s) from LM Studio repository", model_name, quant)

    def on_progress(update: Any) -> None:
        if not logger.isEnabledFor(logging.INFO):
            return
        downloaded_bytes = int(getattr(update, "downloaded_bytes", 0))
        total_bytes = int(getattr(update, "total_bytes", 0))
        logger.info(
            "LM Studio download %s: %d/%d bytes",
            model_name,
            downloaded_bytes,
            total_bytes,
        )

    try:
        model_key = await option.download(on_progress=on_progress)
    except Exception as exc:
        msg = f"failed to download LM Studio model for {model_id!r}"
        raise BackendUnavailableError(msg) from exc
    logger.info("Downloaded LM Studio model %r", model_key)
    return str(model_key)
