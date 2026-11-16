"""Pairwise interactions of an explained input, scored through the pipeline.

@domain pipeline
@see [[knowledge/architecture/pipeline]]
@tested-by [[tests/pipeline/test_explain_interactions.py]]
"""

import itertools
from dataclasses import dataclass

import numpy as np

from ..domain.conversation import ConversationSnapshot
from ..estimation.interactions import pairwise_interactions
from ..plugins import PluginGroup, PluginRegistry, default_registry
from ..protocols.backend import GenerativeBackend
from .config import BudgetConfig, ExplainConfig
from .faithfulness import PlanSampler
from .runner import ExplainRunner


@dataclass(frozen=True, slots=True)
class PairwiseInteractions:
    """Order-2 interaction values between the players of one run."""

    index: str
    """Interaction index, such as ``"k-SII"``."""

    player_ids: tuple[str, ...]
    """Players in matrix order."""

    labels: tuple[str, ...]
    """Text of each player in matrix order; the player ids when the players have no
    spans."""

    values: tuple[tuple[float, ...], ...]
    """Symmetric matrix; entry ``[i][j]`` is the interaction of players ``i`` and ``j``
    and the diagonal is zero. Positive means the pair is worth more together."""

    def top(self, k: int = 5) -> list[tuple[str, str, float]]:
        """Return the ``k`` largest-magnitude pairs as ``(player, player, value)``."""
        pairs = [
            (self.labels[i], self.labels[j], self.values[i][j])
            for i, j in itertools.combinations(range(len(self.player_ids)), 2)
        ]
        return sorted(pairs, key=lambda pair: -abs(pair[2]))[:k]


async def explain_interactions(
    config: ExplainConfig,
    snapshot: ConversationSnapshot,
    index: str = "k-SII",
    registry: PluginRegistry | None = None,
    backend: GenerativeBackend | None = None,
) -> PairwiseInteractions:
    """Score every coalition of ``snapshot`` and return its pairwise interactions.

    All ``2^n`` coalitions run through the scheduler, so an archive or a replay backend
    answers repeats without the model; ``explanation.max_players`` still applies.
    """

    def every_coalition(n: int) -> list[tuple[bool, ...]]:
        return [tuple(bool((key >> i) & 1) for i in range(n)) for key in range(1 << n)]

    sampler = PlanSampler(every_coalition)
    plugins = (registry or default_registry()).copy()
    plugins.register(PluginGroup.ESTIMATORS, sampler.name, lambda _context: sampler)
    explanation = config.explanation.model_copy(
        update={"estimator": sampler.name, "budget": BudgetConfig(fraction=1.0)}
    )
    run_config = config.model_copy(update={"explanation": explanation})
    output = await ExplainRunner(run_config, registry=plugins, backend=backend).explain(
        snapshot
    )
    player_set = output.player_set
    labels = (
        tuple(span.text for span in player_set.spans)
        if player_set is not None and player_set.spans
        else output.manifest.player_ids
    )
    masks = list(sampler.payoffs)
    matrix = pairwise_interactions(masks, [sampler.payoffs[m] for m in masks], index)
    return PairwiseInteractions(
        index=index,
        player_ids=output.manifest.player_ids,
        labels=labels,
        values=tuple(tuple(float(v) for v in row) for row in np.asarray(matrix)),
    )
