"""Practical-test plumbing: feature selection, device, model pool, results file.

``--tier`` (quick, deep or all; default all) and ``--feature`` (``all`` or
comma-separated globs) select over the ``practical`` marker's feature id; the deep
tier runs last, on LM Studio, so each in-process model loads once. Each run writes
``results/<utc>-<tier>-<feature>.json`` and ``results/latest.json`` (git-ignored).
"""

import json
import os
import re
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.practical._cache import STATS
from tests.practical._harness import (
    RUN_INFO,
    ModelPool,
    PracticalModel,
    feature_selected,
    tier_selected,
    vram_gb,
)

RESULTS: dict[str, dict[str, Any]] = {}
"""Per test node id: feature, outcome, seconds and what the test recorded."""

_RECORD = pytest.StashKey[dict[str, Any]]()
_STARTED = time.perf_counter()
RESULTS_DIR = Path(__file__).parent / "results"
"""Git-ignored folder of the results files."""

CPU_SKIP = (
    "practical tests need a GPU; set MLLM_SHAP_PRACTICAL_CPU=1 to run anyway (slow)"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--feature`` and ``--tier``."""
    parser.addoption(
        "--feature",
        default="all",
        help="practical feature ids to run: all, or comma-separated globs",
    )
    parser.addoption(
        "--tier",
        default="all",
        choices=("quick", "deep", "all"),
        help="practical tier: quick, deep (the deep.* ids) or all",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect practical tests outside ``--tier`` and ``--feature``; deep runs last."""
    spec = config.getoption("--feature", default="all")
    tier = config.getoption("--tier", default="all")
    keep, drop = [], []
    for item in items:
        marker = item.get_closest_marker("practical")
        selected = marker is None or (
            feature_selected(spec, marker.args[0])
            and tier_selected(tier, marker.args[0])
        )
        (keep if selected else drop).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
    keep.sort(key=lambda item: not tier_selected("quick", _feature(item)))
    items[:] = keep


def _feature(item: pytest.Item) -> str:
    marker = item.get_closest_marker("practical")
    return "" if marker is None else str(marker.args[0])


@pytest.fixture(scope="session")
def practical_device() -> str:
    """Return ``cuda`` or ``mps``; skip on CPU unless MLLM_SHAP_PRACTICAL_CPU=1."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    elif os.environ.get("MLLM_SHAP_PRACTICAL_CPU") == "1":
        device = "cpu"
    else:
        pytest.skip(CPU_SKIP)
    torch.manual_seed(0)
    RUN_INFO["device"] = device
    return device


@pytest.fixture(scope="session")
def practical_models(request: pytest.FixtureRequest) -> Iterator[ModelPool]:
    """Return the session's model pool; models load on first use.

    The device resolves only for in-process tiers: the deep tier runs in LM Studio.
    """
    pool = ModelPool(lambda: request.getfixturevalue("practical_device"))
    yield pool
    pool.close()


@pytest.fixture
def default_model(practical_models: ModelPool) -> PracticalModel:
    """Return the generator-and-judge tier."""
    return practical_models.get("default")


@pytest.fixture
def fast_model(practical_models: ModelPool) -> PracticalModel:
    """Return the fast tier for mechanics-only scenarios."""
    return practical_models.get("fast")


@pytest.fixture
def deep_model(practical_models: ModelPool) -> PracticalModel:
    """Return the deep tier: an LM Studio model, skipped when the server is down."""
    return practical_models.get("deep")


@pytest.fixture
def deep_hf_model(practical_models: ModelPool) -> PracticalModel:
    """Return in-process Qwen2.5-3B (KV cache); loading it releases the quick tiers."""
    return practical_models.get("deep_hf")


@pytest.fixture
def practical_record(request: pytest.FixtureRequest) -> Iterator[dict[str, Any]]:
    """Return a dict the test fills with properties, numbers and judge margins."""
    record: dict[str, Any] = {"properties": {}, "judge": {}}
    before = STATS.copy()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    yield record
    record["cache"] = dict(STATS - before)
    record.update(vram_gb())
    request.node.stash[_RECORD] = record


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Iterator[None]:
    """Collect each practical test's outcome, duration and record."""
    report: pytest.TestReport = yield
    marker = item.get_closest_marker("practical")
    if marker is None:
        return report
    entry = RESULTS.setdefault(
        item.nodeid, {"feature": marker.args[0], "outcome": "passed", "seconds": 0.0}
    )
    entry["seconds"] = round(entry["seconds"] + report.duration, 3)
    if report.skipped:
        entry["outcome"] = "skipped"
        longrepr = report.longrepr
        entry["reason"] = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
    elif report.failed:
        entry["outcome"] = "failed"
    if report.when == "teardown":
        entry.update(item.stash.get(_RECORD, {}))
        uncertain = any(
            v.get("outcome") == "uncertain" for v in entry.get("judge", {}).values()
        )
        if entry["outcome"] == "passed" and uncertain:
            entry["outcome"] = "uncertain"
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write the results file when practical tests ran."""
    if not RESULTS:
        return
    feature = session.config.getoption("--feature", default="all")
    tier = session.config.getoption("--tier", default="all")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    peaks = [v.get("driver_gb", v.get("vram_gb", 0.0)) for v in RESULTS.values()]
    payload = {
        "utc": stamp,
        "tier": tier,
        "feature": feature,
        **RUN_INFO,
        "wall_seconds": round(time.perf_counter() - _STARTED, 1),
        "peak_vram_gb": max(peaks, default=0.0),
        "cache": dict(STATS),
        "scenarios": RESULTS,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"{stamp}-{tier}-{re.sub(r'[^A-Za-z0-9._-]+', '_', feature)}.json"
    text = json.dumps(payload, indent=2) + "\n"
    for target in (name, "latest.json"):
        (RESULTS_DIR / target).write_text(text, encoding="utf-8")
