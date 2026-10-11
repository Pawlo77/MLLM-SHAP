"""Deep-tier helpers: a one-message token partition, cached references, sweeps.

An exact reference is kept in ``results/reference-<hash>.json``; the hash covers
the model pin, the device, the rendered chat and the config, so a cold generation
cache reuses it unless one of them changed.
"""

import itertools
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from pydantic import Field

from mllm_shap.backends._chat import snapshot_to_chat_messages
from mllm_shap.domain.conversation import ConversationSnapshot
from mllm_shap.domain.players import PlayerSet
from mllm_shap.masking.tokens import token_spans
from mllm_shap.plugins import PluginContext, PluginGroup, default_registry
from mllm_shap.protocols.plugin import PluginOptions

from ._cache import cache_key
from ._harness import HERE, RUN_INFO, PracticalModel, Scenario, labels
from ._judge import normalised_l1


class FocusOptions(PluginOptions):
    """Options of the ``focus_tokens`` partition."""

    message_index: int = Field(default=0, ge=0)
    """Message, in conversation order (turns flattened), whose tokens are players."""


class FocusTokens:
    """Whitespace tokens of one message are the players; other text stays."""

    options_model: ClassVar[type[PluginOptions]] = FocusOptions

    def __init__(self, context: PluginContext | None = None) -> None:
        """Read option ``message_index``."""
        self.message_index = 0
        if context is not None:
            self.message_index = context.option("message_index", 0)

    @property
    def name(self) -> str:
        """Return the registered partition identifier."""
        return "focus_tokens"

    def partition(self, snapshot: ConversationSnapshot) -> PlayerSet:
        """Keep the token players of the chosen message."""
        pairs = [
            pair
            for pair in token_spans(snapshot)
            if pair[1].message_index == self.message_index
        ]
        return PlayerSet(
            player_ids=tuple(p for p, _ in pairs), spans=tuple(s for _, s in pairs)
        )


REGISTRY = default_registry()
"""Built-in plugins plus ``focus_tokens``."""
REGISTRY.register(PluginGroup.PARTITIONS, "focus_tokens", FocusTokens)


def reference(
    model: PracticalModel, scenario: Scenario, variant: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the exact run's labels and values, from its file when present."""
    config = scenario.config(model.spec, variant)
    key = cache_key(
        RUN_INFO["device"],
        snapshot_to_chat_messages(scenario.snapshot),
        config.model_dump(mode="json"),
    )
    path = HERE / "results" / f"reference-{key[:16]}.json"
    if path.exists():
        stored: dict[str, Any] = json.loads(path.read_text("utf-8"))
        return stored
    started = time.perf_counter()
    output = model.explain(scenario, variant)
    data = {
        "labels": labels(output),
        "values": list(output.result.values),
        "base_value": output.result.base_value,
        "grand_value": output.result.grand_value,
        "executed": output.metrics.executed,
        "seconds": round(time.perf_counter() - started, 1),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", "utf-8")
    return data


def sweep(
    model: PracticalModel,
    scenario: Scenario,
    budgets: Sequence[int],
    seeds: Sequence[int],
    variant: Mapping[str, Any] | None = None,
) -> dict[int, list[list[float]]]:
    """Return the values of each budget and seed, on top of ``variant``."""
    runs: dict[int, list[list[float]]] = {}
    for budget in budgets:
        runs[budget] = []
        for seed in seeds:
            explanation = dict((variant or {}).get("explanation", {}))
            explanation.update(budget={"evaluations": budget}, seed=seed)
            output = model.explain(scenario, {"explanation": explanation})
            runs[budget].append(list(output.result.values))
    return runs


def error_table(
    runs: Mapping[str, Mapping[int, Sequence[Sequence[float]]]],
    reference: Sequence[float],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Return the normalised L1 error per seed and its median, by method and budget."""
    table: dict[str, dict[int, dict[str, Any]]] = {}
    for method, by_budget in runs.items():
        table[method] = {}
        for budget, values in by_budget.items():
            errors = [round(normalised_l1(v, reference), 4) for v in values]
            table[method][budget] = {
                "per_seed": errors,
                "median": statistics.median(errors),
            }
    return table


def non_increasing(values: Sequence[float]) -> bool:
    """Return whether each value is at most the one before it."""
    return all(b <= a for a, b in itertools.pairwise(values))


def top1_agreement(
    runs: Sequence[Sequence[float]], reference: Sequence[float]
) -> float:
    """Return the share of runs whose largest value is the reference's top player."""
    best = max(range(len(reference)), key=lambda i: reference[i])
    hits = sum(max(range(len(v)), key=lambda i: v[i]) == best for v in runs)
    return hits / len(runs)
