"""Pairwise interaction indices of a full coalition table, through shapiq.

The table is wrapped as a ``shapiq`` game and ``ExactComputer`` returns the order-2
index: ``k-SII`` (the Shapley interaction index at its top order, the default),
``SII``, ``FSII`` (Faith-SHAP) or any other index it accepts. Sampled tables are not
supported, because the approximators of ``shapiq`` choose their own coalitions.

@domain estimation
@see [[knowledge/architecture/estimation]]
@tested-by [[tests/estimation/test_interactions.py]]
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._shared import present_to_mask_int


def pairwise_interactions(
    masks: Sequence[Sequence[bool]],
    payoffs: Sequence[float],
    index: str = "k-SII",
) -> np.ndarray:
    """Return the symmetric matrix of order-2 ``index`` values, zero on the diagonal.

    Raises:
        ImportError: If ``shapiq`` is not installed (extra ``interactions``).
        ValueError: If ``masks`` is not the complete table of ``2^n`` coalitions.
    """
    try:
        import shapiq
    except ImportError as error:
        msg = (
            "pairwise interactions need shapiq; install it with "
            "'mllm-shap[interactions]'"
        )
        raise ImportError(msg) from error
    num_players = len(masks[0])
    table = np.full(1 << num_players, np.nan)
    for mask, payoff in zip(masks, payoffs, strict=True):
        table[present_to_mask_int(mask)] = payoff
    if np.isnan(table).any():
        msg = "interactions need the complete table of 2^n coalitions"
        raise ValueError(msg)

    class _TableGame(shapiq.Game):
        def __init__(self) -> None:
            super().__init__(n_players=num_players, normalize=False, verbose=False)

        def value_function(self, coalitions: np.ndarray) -> np.ndarray:
            keys = (coalitions.astype(np.int64) << np.arange(num_players)).sum(axis=1)
            values: np.ndarray = table[keys]
            return values

    computer: Any = shapiq.ExactComputer(n_players=num_players, game=_TableGame())
    computed = computer(index, order=2)
    matrix: np.ndarray = np.asarray(computed.get_n_order_values(2), dtype=float)
    return matrix
