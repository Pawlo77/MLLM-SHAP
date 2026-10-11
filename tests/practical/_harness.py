"""Scenario catalogue, model pool and run helpers of the practical tests.

``scenarios.yaml`` and ``models.yaml`` are the single source; tests take their
snapshots, configs and thresholds from here and run the explain pipeline on a
session-wide model whose generations go through the SQLite cache.
"""

import asyncio
import dataclasses
import fnmatch
import gc
import json
import os
import subprocess  # nosec B404 - fixed argv
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from mllm_shap import ExplainConfig, ExplainRunner, explain_config_to_yaml
from mllm_shap.backends.api import ApiBackend
from mllm_shap.backends.transformers.text import TransformersTextBackend
from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.pipeline.config import BackendConfig
from mllm_shap.pipeline.result import ExplainRunOutput
from mllm_shap.plugins import PluginContext, PluginRegistry
from mllm_shap.protocols.backend import GenerativeBackend

from ._cache import DEFAULT_PATH, CachedBackend, PracticalCache
from ._judge import Judge, LmStudioJudge, LogitJudge

HERE = Path(__file__).parent

RUN_INFO: dict[str, Any] = {"device": None, "models": {}, "judges": {}}
"""Device, loaded model pins and judges, written into the results file."""

LOOP = asyncio.new_event_loop()
"""One loop for the whole session: an HTTP client cannot outlive ``asyncio.run``."""

LMS_HOST = "http://localhost:1234/v1"
"""LM Studio's OpenAI-compatible server, the backend of ``kind: api`` tiers."""


@dataclass(frozen=True)
class ModelSpec:
    """One tier of ``models.yaml``: a pinned transformers model or an LM Studio key."""

    model_id: str
    size_gb: float
    kind: str = "transformers"
    revision: str = ""
    dtype: str = ""

    def backend_config(self) -> dict[str, Any]:
        """Return the ``backend`` section of an ``api`` or transformers config."""
        if self.kind == "api":
            return {"kind": "api", "model_id": self.model_id, "api_host": LMS_HOST}
        options = {"revision": self.revision, "dtype": self.dtype}
        return {"kind": "transformers", "model_id": self.model_id, "options": options}


@dataclass(frozen=True)
class Scenario:
    """One entry of ``scenarios.yaml``."""

    id: str
    tier: str
    turns: list[list[dict[str, str]]]
    overrides: dict[str, Any] = field(default_factory=dict)
    variants: list[dict[str, Any]] = field(default_factory=list)
    expect: dict[str, Any] = field(default_factory=dict)

    @property
    def snapshot(self) -> ConversationSnapshot:
        """Return the scenario conversation."""
        return ConversationSnapshot.from_turns(
            tuple(
                Turn(messages=tuple(_message(m) for m in turn)) for turn in self.turns
            )
        )

    def config(
        self, model: ModelSpec, variant: Mapping[str, Any] | None = None
    ) -> ExplainConfig:
        """Return defaults, overrides and ``variant`` merged, for ``model``."""
        data = _merge({"backend": model.backend_config(), **DEFAULTS}, self.overrides)
        return ExplainConfig.model_validate(_merge(data, variant or {}))


def _message(payload: Mapping[str, str]) -> Message:
    return Message(
        role=Role(payload["role"]), text=payload["text"], speaker=payload.get("speaker")
    )


def _merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge ``update`` into ``base``; a ``None`` value deletes the key."""
    merged = dict(base)
    for key, value in update.items():
        if value is None:
            merged.pop(key, None)
            continue
        old = merged.get(key)
        both = isinstance(old, Mapping) and isinstance(value, Mapping)
        merged[key] = _merge(old, value) if both else value
    return merged


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((HERE / name).read_text(encoding="utf-8"))
    return data


MODELS = {tier: ModelSpec(**spec) for tier, spec in _load("models.yaml").items()}
"""Model tiers: ``default`` (generator and judge), ``fast``, ``deep`` (LM Studio)."""

_CATALOGUE = _load("scenarios.yaml")
DEFAULTS: dict[str, Any] = _CATALOGUE["defaults"]
"""Config sections every scenario starts from."""

SCENARIOS = {
    sid: Scenario(id=sid, **entry) for sid, entry in _CATALOGUE["scenarios"].items()
}
"""Scenarios by feature id."""


def feature_selected(spec: str, feature: str) -> bool:
    """Return whether ``feature`` matches ``all`` or one comma-separated glob."""
    globs = [g.strip() for g in spec.split(",") if g.strip()]
    return any(g == "all" or fnmatch.fnmatchcase(feature, g) for g in globs)


def tier_selected(spec: str, feature: str) -> bool:
    """Return whether ``feature`` is in run tier ``spec`` (quick, deep or all)."""
    return spec == "all" or spec == ("deep" if feature.startswith("deep.") else "quick")


def scenario_cases(*ids: str) -> list[Any]:
    """Return ``pytest.param`` per scenario, marked ``practical(<id>)`` and named so."""
    return [
        pytest.param(SCENARIOS[i], id=i, marks=pytest.mark.practical(i)) for i in ids
    ]


@dataclass
class PracticalModel:
    """A loaded tier: the raw backend, its cached view and the judge."""

    spec: ModelSpec
    backend: TransformersTextBackend | ApiBackend
    cached: CachedBackend
    judge: Judge
    identity: dict[str, str]
    """Model, revision and device that key the generation cache and references."""

    def explain(
        self,
        scenario: Scenario,
        variant: Mapping[str, Any] | None = None,
        backend: GenerativeBackend | None = None,
        registry: PluginRegistry | None = None,
    ) -> ExplainRunOutput:
        """Run ``scenario`` through the cache (or ``backend``) without closing it."""
        runner = ExplainRunner(
            scenario.config(self.spec, variant),
            backend=backend or self.cached,
            registry=registry,
        )
        return LOOP.run_until_complete(runner.explain(scenario.snapshot))

    def reanalyze(
        self,
        scenario: Scenario,
        archive: Path,
        variant: Mapping[str, Any] | None = None,
    ) -> ExplainRunOutput:
        """Re-score ``archive`` with the scenario config; no backend is built."""
        return ExplainRunner(scenario.config(self.spec, variant)).reanalyze(archive)

    def answer(self, scenario: Scenario) -> str:
        """Return the base generation of ``scenario`` (a cache hit after a run)."""
        config = scenario.config(self.spec)
        record = LOOP.run_until_complete(
            self.cached.generate(scenario.snapshot, config.generation)
        )
        return record.text


def archived(path: Path) -> dict[str, Any]:
    """Return the config variant that writes the run archive to ``path``."""
    archive = {"path": str(path), "on_exists": "overwrite"}
    return {"explanation": {"archive": archive}}


def cli_explain(scenario: Scenario, model: PracticalModel, folder: Path) -> Any:
    """Run ``mllm-shap explain`` in a subprocess; return the result JSON."""
    config = scenario.config(model.spec, archived(folder / "run"))
    (folder / "config.yaml").write_text(explain_config_to_yaml(config), "utf-8")
    (folder / "snapshot.json").write_text(scenario.snapshot.to_json(), "utf-8")
    run_cli(
        "explain",
        "--config",
        str(folder / "config.yaml"),
        "--snapshot",
        str(folder / "snapshot.json"),
        "--out",
        str(folder / "out.json"),
        "--render",
        "token_bar",
        "--png",
        str(folder / "figure.png"),
    )
    return json.loads((folder / "out.json").read_text("utf-8"))


def run_cli(*args: str) -> None:
    """Run ``python -m mllm_shap.cli`` offline unless downloads are allowed."""
    env = dict(os.environ)
    if env.get("MLLM_SHAP_PRACTICAL_DOWNLOAD") != "1":
        env["HF_HUB_OFFLINE"] = "1"
    done = subprocess.run(  # nosec B603
        [sys.executable, "-m", "mllm_shap.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-2000:]


def labels(output: ExplainRunOutput) -> list[str]:
    """Return each player's display label, in value order."""
    return [str(label["label"]) for label in output.to_dict()["labels"]]


def summary(output: ExplainRunOutput) -> dict[str, Any]:
    """Return the numbers a results file keeps for one run."""
    return {
        "labels": labels(output),
        "values": [round(v, 6) for v in output.result.values],
        "base_value": round(output.result.base_value, 6),
        "grand_value": round(output.result.grand_value, 6),
        "executed": output.metrics.executed,
    }


class ModelPool:
    """Load each tier once per session, on first use; ``deep_hf`` owns the device."""

    def __init__(self, resolve_device: Callable[[], str]) -> None:
        """Resolve the device lazily (LM Studio tiers need none); share one cache."""
        self._resolve_device = resolve_device
        self.cache = PracticalCache(DEFAULT_PATH)
        self._models: dict[str, PracticalModel] = {}

    def get(self, tier: str) -> PracticalModel:
        """Return the tier's model, skipping when its weights or server are missing."""
        if tier not in self._models:
            spec = MODELS[tier]
            if spec.kind == "api":
                self._models[tier] = self._load_lmstudio(spec)
            else:
                self._models[tier] = self._load_transformers(tier, spec)
            RUN_INFO["models"][tier] = f"{spec.model_id}@{spec.revision or spec.kind}"
        return self._models[tier]

    def _load_transformers(self, tier: str, spec: ModelSpec) -> PracticalModel:
        require_weights(spec)
        if tier == "deep_hf" or "deep_hf" in self._models:
            self._close_in_process()
        device = self._resolve_device()
        backend = TransformersTextBackend(_context(spec))
        identity = {
            "model": spec.model_id,
            "revision": spec.revision,
            "device": device,
            "dtype": spec.dtype,
        }
        judge: Judge = LogitJudge(backend, self.cache, identity)
        if tier == "default" and (key := _judge_key()):
            judge = LmStudioJudge(key, self.cache)
        RUN_INFO["judges"][tier] = judge.name
        return PracticalModel(
            spec, backend, CachedBackend(backend, self.cache, identity), judge, identity
        )

    def _load_lmstudio(self, spec: ModelSpec) -> PracticalModel:
        key = os.environ.get("MLLM_SHAP_PRACTICAL_DEEP_MODEL")
        spec = dataclasses.replace(spec, model_id=key) if key else spec
        info = require_lmstudio(spec)
        judge_key = os.environ.get("MLLM_SHAP_PRACTICAL_DEEP_JUDGE")
        judge_spec = MODELS["deep_judge"]
        if judge_key:
            judge_spec = dataclasses.replace(judge_spec, model_id=judge_key)
        require_lmstudio(judge_spec)
        backend = ApiBackend(_context(spec))
        identity = {
            "model": spec.model_id,
            "publisher": str(info.get("publisher")),
            "quantization": str(info.get("quantization")),
            "device": "lmstudio",
        }
        RUN_INFO["device"] = RUN_INFO["device"] or "lmstudio"
        judge = LmStudioJudge(judge_spec.model_id, self.cache)
        RUN_INFO["judges"]["deep"] = judge.name
        return PracticalModel(
            spec, backend, CachedBackend(backend, self.cache, identity), judge, identity
        )

    def _close_in_process(self) -> None:
        for tier, model in list(self._models.items()):
            if model.spec.kind == "transformers":
                self.release(tier)

    def release(self, tier: str) -> None:
        """Drop ``tier``'s backend and return the memory to the device."""
        model = self._models.pop(tier, None)
        if model is None:
            return
        LOOP.run_until_complete(model.backend.aclose())
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Release every loaded model."""
        for tier in list(self._models):
            self.release(tier)


def _context(spec: ModelSpec) -> PluginContext:
    backend = BackendConfig.model_validate(spec.backend_config())
    return PluginContext(config=ExplainConfig(backend=backend))


def _judge_key() -> str:
    """Return the LM Studio key of ``MLLM_SHAP_PRACTICAL_JUDGE=lmstudio:<key>``."""
    value = os.environ.get("MLLM_SHAP_PRACTICAL_JUDGE", "")
    return value.removeprefix("lmstudio:") if value.startswith("lmstudio:") else ""


def vram_gb() -> dict[str, float]:
    """Return device memory in GB: CUDA peak, or MPS allocated and driver totals."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    if torch.cuda.is_available():
        return {"vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if torch.backends.mps.is_available():
        return {
            "vram_gb": round(torch.mps.current_allocated_memory() / 1e9, 2),
            "driver_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2),
        }
    return {}


WEIGHT_FILES = ("*.json", "*.safetensors", "*.txt", "*.model")
"""Files transformers loads; it never fetches README, LICENSE or .gitattributes."""


def require_weights(spec: ModelSpec) -> None:
    """Skip unless the pinned weights are cached or downloads are allowed."""
    if os.environ.get("MLLM_SHAP_PRACTICAL_DOWNLOAD") == "1":
        return
    hub = pytest.importorskip("huggingface_hub")
    try:
        hub.snapshot_download(
            spec.model_id,
            revision=spec.revision,
            local_files_only=True,
            allow_patterns=list(WEIGHT_FILES),
        )
    except hub.errors.LocalEntryNotFoundError:
        pytest.skip(
            f"{spec.model_id}@{spec.revision[:8]} is not in the HF cache; run "
            f"`huggingface-cli download {spec.model_id} --revision {spec.revision}` "
            "or set MLLM_SHAP_PRACTICAL_DOWNLOAD=1"
        )


def require_lmstudio(spec: ModelSpec) -> dict[str, Any]:
    """Skip unless LM Studio serves ``spec.model_id``; return its model info."""
    httpx = pytest.importorskip("httpx")
    root = LMS_HOST.removesuffix("/v1")
    try:
        response = httpx.get(f"{root}/api/v0/models/{spec.model_id}", timeout=3.0)
        info: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError) as error:
        pytest.skip(
            f"LM Studio is not serving at {LMS_HOST} (lms server start): {error}"
        )
    if response.status_code != 200 or info.get("type") not in {"llm", "vlm"}:
        pytest.skip(
            f"LM Studio has no chat model {spec.model_id!r} (download: lms get)"
        )
    return info
