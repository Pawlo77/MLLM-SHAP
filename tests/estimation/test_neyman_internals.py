"""Known-truth tests for the Neyman estimator's internal steps.

Each step is driven in isolation on hand-built state: phase-one sizing, the
cell targets, the round-robin cursor, the stratum-variance estimate and the
water-filling allocation, and the error contracts. A pinned seeded run covers
the order in which the steps are combined.

@see [[src/nlp_shap/estimation/neyman.py]]
"""

import numpy as np
import pytest

from nlp_shap.domain.players import PlayerSet
from nlp_shap.estimation._shared import present_to_mask_int
from nlp_shap.estimation.neyman import NeymanEstimator, _NeymanStep
from nlp_shap.pipeline.config import BackendConfig, ExplainConfig
from nlp_shap.plugins import PluginContext


def _players(num_players: int) -> PlayerSet:
    return PlayerSet(player_ids=tuple(f"p{i}" for i in range(num_players)))


def _payoff(present: tuple[bool, ...]) -> float:
    """Non-additive game: |S|^2 plus 3 when the first and last players meet."""
    return float(sum(present) ** 2 + (3.0 if present[0] and present[-1] else 0.0))


def test_option_defaults() -> None:
    estimator = NeymanEstimator()
    assert estimator.initial_fraction is None
    assert estimator.initial_num_samples is None
    assert estimator.use_standard_method is False
    assert estimator._use_default_initial_formula is True


def test_options_come_from_the_context() -> None:
    config = ExplainConfig.model_validate({
        "backend": BackendConfig(kind="mock", model_id="mock").model_dump(),
        "explanation": {
            "neyman": {
                "initial_fraction": 0.25,
                "initial_num_samples": 5,
                "use_standard_method": True,
            }
        },
    })
    estimator = NeymanEstimator(
        PluginContext(config=config),
        initial_fraction=0.9,
        initial_num_samples=9,
        use_standard_method=False,
    )
    assert estimator.initial_fraction == 0.25
    assert estimator.initial_num_samples == 5
    assert estimator.use_standard_method is True
    assert estimator._use_default_initial_formula is False


@pytest.mark.parametrize(
    ("options", "num_players", "total", "expected"),
    [
        # Default: max(2, ceil(total / (2 n^2))).
        ({}, 10, 1024, 6),  # ceil(1024 / 200) = ceil(5.12)
        ({}, 10, 400, 2),  # 400 / 200 = 2 exactly
        ({}, 10, 401, 3),  # ceil(2.005)
        ({}, 3, 8, 2),  # ceil(8 / 18) = 1, raised to 2
        ({"initial_num_samples": 7}, 10, 1024, 7),
        ({"initial_num_samples": 1}, 10, 1024, 2),
        # initial_fraction is a complementary budget: 0.5 of 2^6 = 32 masks.
        ({"initial_fraction": 0.5}, 6, 64, 32),
        ({"initial_fraction": 0.01}, 6, 64, 14),  # floor 2n + 2
        # initial_num_samples wins over initial_fraction.
        ({"initial_fraction": 0.5, "initial_num_samples": 4}, 6, 64, 4),
    ],
)
def test_resolve_initial_num_splits(
    options: dict[str, float], num_players: int, total: int, expected: int
) -> None:
    estimator = NeymanEstimator(**options)  # type: ignore[arg-type]  # test kwargs
    assert estimator._resolve_initial_num_splits(num_players, total) == expected


def test_cell_targets_cap_by_feasible_coalitions() -> None:
    estimator = NeymanEstimator(initial_num_samples=3)
    estimator._reset_neyman_state(4, 1.0, seed=0)
    # Column k holds coalitions of size k containing the player: C(3, k - 1) of
    # them (1 for k = 0): min(3, [1, 1, 3, 3, 1]).
    assert estimator._cell_targets().tolist() == [[1, 1, 3, 3, 1]] * 4


@pytest.mark.parametrize(("num_players", "expected"), [(1, 1), (5, 3), (6, 3)])
def test_allocation_starts_at_the_larger_half(num_players: int, expected: int) -> None:
    estimator = NeymanEstimator()
    estimator.reset_sampling_state(num_players)
    assert estimator._allocation_start_index() == expected


def _cursor_estimator() -> NeymanEstimator:
    """Three players, fresh state: targets [1, 1, 2, 1] per player, cursor (1, 0)."""
    estimator = NeymanEstimator()
    estimator._reset_neyman_state(3, 1.0, seed=0)
    assert estimator._cell_targets().tolist() == [[1, 1, 2, 1]] * 3
    assert (estimator._position_j, estimator._position_i) == (1, 0)
    return estimator


def _walk(estimator: NeymanEstimator, steps: int) -> list[tuple[int, int, bool]]:
    visited = []
    for _ in range(steps):
        done = estimator._update_m_position()
        visited.append((estimator._position_j, estimator._position_i, done))
    return visited


def test_cursor_walks_size_by_size_round_robin_over_players() -> None:
    # Cells (size, player) in size-major order; sizes 0 and n = 3 are skipped,
    # and the walk wraps around to the first incomplete cell.
    assert _walk(_cursor_estimator(), 7) == [
        (1, 1, False),
        (1, 2, False),
        (2, 0, False),
        (2, 1, False),
        (2, 2, False),
        (1, 0, False),
        (1, 1, False),
    ]


def test_cursor_skips_complete_cells() -> None:
    estimator = _cursor_estimator()
    m_counts = estimator._m_counts
    assert m_counts is not None
    m_counts[:, 1] = 1  # size 1 complete for everyone
    m_counts[1, 2] = 2  # size 2 complete for player 1
    estimator._position_j, estimator._position_i = 1, 2
    assert _walk(estimator, 3) == [(2, 0, False), (2, 2, False), (2, 0, False)]


def test_cursor_reports_done_when_inner_cells_are_complete() -> None:
    estimator = _cursor_estimator()
    m_counts = estimator._m_counts
    assert m_counts is not None
    # Empty (size 0) and grand (size 3) columns never count as incomplete.
    m_counts[:, 1] = 1
    m_counts[:, 2] = 2
    assert estimator._update_m_position() is True
    m_counts[2, 2] = 1  # one inner cell open again
    assert estimator._update_m_position() is False
    assert (estimator._position_j, estimator._position_i) == (2, 2)


def test_cursor_requires_sampling_state() -> None:
    with pytest.raises(
        RuntimeError, match=r"^M matrix must be initialized before sampling\.$"
    ):
        NeymanEstimator()._update_m_position()


def _allocation_estimator() -> NeymanEstimator:
    """Six players with hand-picked phase-one sums and 12 seen coalitions.

    Per cell (player, size): M observations, C = sum of contributions and
    C2 = sum of squares; sigma^2 = (C2 - C^2 / M) / (M - 1), set to 0 when
    M < 2 and clipped at 0.

    - size 1: player 1, M=3, C=0, C2=0.5 -> 0.25; player 2, M=1 -> 0. Sum 1/4.
    - size 2: player 3, M=3, C=0, C2=0.125 -> 1/16; player 4, M=2, C=0.75,
      C2=0.3125 -> (0.3125 - 0.28125) / 1 = 1/32. Sum 3/32.
    - size 3: player 0, M=2, C=0.75, C2=0.3125 -> 1/32; player 1, M=2, C=1,
      C2=0 -> -0.5, clipped to 0. Sum 1/32.
    - size 4: player 0, M=2, C=1, C2=0.625 -> 1/8; player 2, M=2, C=0.75,
      C2=0.3125 -> 1/32. Sum 5/32.
    - size 5: player 5, M=2, C=1, C2=1 -> 1/2; player 0, M=3, C=0, C2=0.5 ->
      1/4. Sum 3/4.

    Phase two samples the larger side k of a pair, k = 3, 4, 5, paired with
    the complement size 6 - k. The Neyman weight is
    sqrt(sum sigma^2[:, k] / (k + 1) + sum sigma^2[:, 6 - k] / (7 - k)):

    - k = 3: sqrt((1/32) / 4 + (1/32) / 4) = 1/8
    - k = 4: sqrt((5/32) / 5 + (3/32) / 3) = 1/4
    - k = 5: sqrt((3/4) / 6 + (1/4) / 2) = 1/2

    Unseen pairs: size 3 has (C(6, 3) - 4 seen) / 2 = 8 (both sides have size
    3), size 4 has C(6, 4) - 3 = 12 and size 5 has C(6, 5) - 5 = 1.
    """
    cells = {
        (1, 1): (3, 0.0, 0.5),
        (2, 1): (1, 0.5, 0.25),
        (3, 2): (3, 0.0, 0.125),
        (4, 2): (2, 0.75, 0.3125),
        (0, 3): (2, 0.75, 0.3125),
        (1, 3): (2, 1.0, 0.0),
        (0, 4): (2, 1.0, 0.625),
        (2, 4): (2, 0.75, 0.3125),
        (5, 5): (2, 1.0, 1.0),
        (0, 5): (3, 0.0, 0.5),
    }
    size_3 = {0b000111, 0b001011, 0b001101, 0b001110}
    size_4 = {0b111100, 0b111010, 0b111001}
    size_5 = {0b111110, 0b111101, 0b111011, 0b110111, 0b101111}
    # Coalitions outside sizes 3 to 5 do not change the capacity.
    return _six_player_estimator(
        cells, size_3 | size_4 | size_5 | {0, 0b111111, 0b000001}
    )


def _six_player_estimator(
    cells: dict[tuple[int, int], tuple[int, float, float]], seen: set[int]
) -> NeymanEstimator:
    """Set phase-one sums ``(player, size) -> (M, C, C2)`` and the seen masks."""
    estimator = NeymanEstimator()
    estimator._num_players = 6
    m_counts = np.zeros((6, 7), dtype=np.int64)
    c_matrix = np.zeros((6, 7))
    c_squared = np.zeros((6, 7))
    for (player, size), (count, total, squares) in cells.items():
        m_counts[player, size] = count
        c_matrix[player, size] = total
        c_squared[player, size] = squares
    estimator._m_counts = m_counts
    estimator._c_matrix = c_matrix
    estimator._c_squared = c_squared
    estimator._seen = seen
    return estimator


def test_unseen_pairs_by_size() -> None:
    assert _allocation_estimator()._unseen_pairs_by_size().tolist() == [
        0,
        0,
        0,
        8,
        12,
        1,
        0,
    ]


def _m_hat(estimator: NeymanEstimator) -> list[float]:
    assert estimator._m_hat is not None
    return [float(value) for value in estimator._m_hat]


def test_estimate_m_hat_water_fills_by_neyman_weight() -> None:
    # Weights w = (1/8, 1/4, 1/2), sum 7/8; capacity (8, 12, 1); 19 pairs.
    # Pass 1: floor(19 * w / (7/8)) = floor(19/7, 38/7, 76/7) = (2, 5, 10),
    #   capped by capacity -> (2, 5, 1); 11 pairs left.
    # Pass 2: size 5 is full; floor(11 * (1/8, 1/4) / (3/8)) = floor(3.67, 7.33)
    #   = (3, 7) -> (5, 12, 1); 1 pair left.
    # Pass 3: only size 3 has room: floor(1 * 1) = 1 -> (6, 12, 1).
    estimator = _allocation_estimator()
    estimator._estimate_m_hat(19)
    assert _m_hat(estimator) == [0.0, 0.0, 0.0, 6.0, 12.0, 1.0]


def test_estimate_m_hat_weights_follow_stratum_variances() -> None:
    # Nothing seen: capacity (10, 15, 6). Stratum variances:
    # - size 1: player 5, M=1 -> 0 (fewer than two observations).
    # - size 2: player 1, M=2, C=0.5, C2=0.25 -> (0.25 - 0.125) / 1 = 1/8.
    # - size 3: player 2, M=2, C=2, C2=4 -> (4 - 2) / 1 = 2; player 5, M=2,
    #   C=1, C2=0 -> -0.5, clipped to 0.
    # - size 5: player 4, M=2, C=2, C2=4 -> 2; player 5, M=3, C=0, C2=0.5 ->
    #   0.5 / 2 = 1/4.
    # Weights: size 3 sqrt(2/4 + 2/4) = 1; size 4 sqrt(0/5 + (1/8)/3)
    # = sqrt(1/24) ~ 0.2041; size 5 sqrt((9/4)/6 + 0/2) = sqrt(3/8) ~ 0.6124.
    # 9 pairs: floor(9 * w / 1.8165) = floor(4.955, 1.011, 3.034) = (4, 1, 3);
    # the last pair goes to the largest weight (size 3) -> (5, 1, 3).
    estimator = _six_player_estimator(
        {
            (5, 1): (1, 0.5, 0.25),
            (1, 2): (2, 0.5, 0.25),
            (2, 3): (2, 2.0, 4.0),
            (5, 3): (2, 1.0, 0.0),
            (4, 5): (2, 2.0, 4.0),
            (5, 5): (3, 0.0, 0.5),
        },
        set(),
    )
    estimator._estimate_m_hat(9)
    assert _m_hat(estimator) == [0.0, 0.0, 0.0, 5.0, 1.0, 3.0]


def test_estimate_m_hat_is_capped_by_unseen_pairs() -> None:
    estimator = _allocation_estimator()
    estimator._estimate_m_hat(100)
    assert _m_hat(estimator) == [0.0, 0.0, 0.0, 8.0, 12.0, 1.0]


def test_estimate_m_hat_zero_variance_splits_by_capacity() -> None:
    # Weights fall back to the spare capacity (8, 12, 1), sum 21; 5 pairs.
    # Pass 1: floor(5 * (8, 12, 1) / 21) = floor(1.90, 2.86, 0.24) = (1, 2, 0).
    # Pass 2: spare (7, 10, 1), sum 18: floor(2 * spare / 18) = (0, 1, 0).
    # Pass 3: spare (7, 9, 1), sum 17: all shares floor to 0, so the size with
    #   the largest weight (size 4) gets the last pair -> (1, 4, 0).
    estimator = _allocation_estimator()
    assert estimator._c_matrix is not None
    assert estimator._c_squared is not None
    estimator._c_matrix[:] = 0.0
    estimator._c_squared[:] = 0.0
    estimator._estimate_m_hat(5)
    assert _m_hat(estimator) == [0.0, 0.0, 0.0, 1.0, 4.0, 0.0]


@pytest.mark.parametrize("missing", ["_m_counts", "_c_matrix", "_c_squared"])
def test_estimate_m_hat_requires_phase_one_sums(missing: str) -> None:
    estimator = _allocation_estimator()
    setattr(estimator, missing, None)
    with pytest.raises(
        RuntimeError, match=r"^Neyman allocation inputs must be initialized\.$"
    ):
        estimator._estimate_m_hat(1)


def test_begin_allocation_contracts() -> None:
    with pytest.raises(
        RuntimeError, match=r"^M matrix must be initialized before allocation\.$"
    ):
        NeymanEstimator().begin_allocation([], [])
    estimator = NeymanEstimator()
    masks = list(estimator.sample_masks(_players(4), 1.0, False, seed=0))
    payoffs = [_payoff(mask.present) for mask in masks]
    odd = r"^Remaining samples for Neyman allocation must be an even integer\.$"
    # A repeated mask makes the count odd; twice the budget leaves -16 masks.
    with pytest.raises(RuntimeError, match=odd):
        estimator.begin_allocation([*masks, masks[0]], [*payoffs, payoffs[0]])
    with pytest.raises(RuntimeError, match=odd):
        estimator.begin_allocation(masks * 2, payoffs * 2)


def test_sample_allocation_masks_contracts() -> None:
    with pytest.raises(
        RuntimeError, match=r"^Neyman allocation has not been initialized\.$"
    ):
        list(NeymanEstimator().sample_allocation_masks())
    estimator = NeymanEstimator()
    estimator._step = _NeymanStep.NEYMAN_ALLOCATION
    with pytest.raises(
        RuntimeError, match=r"^M_hat must be initialized before allocation sampling\.$"
    ):
        list(estimator.sample_allocation_masks())


def test_sample_masks_requires_no_minimal_masks() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Neyman estimator does not support include_minimal_masks=True$",
    ):
        list(NeymanEstimator().sample_masks(_players(3), 1.0, True, seed=0))


def test_allocation_beyond_unseen_pairs_stops() -> None:
    """An allocation larger than a size's unseen pairs ends once they run out."""
    estimator = NeymanEstimator()
    phase_one = list(estimator.sample_masks(_players(6), 1.0, False, seed=0))
    estimator.begin_allocation(phase_one, [_payoff(m.present) for m in phase_one])
    assert estimator._m_hat is not None
    unseen_size_4 = int(estimator._unseen_pairs_by_size()[4])
    planned_size_3 = int(estimator._m_hat[3])
    estimator._m_hat[4] = unseen_size_4 + 50
    phase_two = list(estimator.sample_allocation_masks())
    assert len(phase_two) == 2 * (planned_size_3 + unseen_size_4)


def _run(
    num_players: int, fraction: float, **options: float | bool
) -> tuple[list[int], list[float], list[int]]:
    estimator = NeymanEstimator(**options)  # type: ignore[arg-type]  # test kwargs
    phase_one = list(estimator.sample_masks(_players(num_players), fraction, False, 0))
    estimator.begin_allocation(phase_one, [_payoff(m.present) for m in phase_one])
    m_hat = _m_hat(estimator)
    phase_two = list(estimator.sample_allocation_masks())
    return (
        [present_to_mask_int(m.present) for m in phase_one],
        m_hat,
        [present_to_mask_int(m.present) for m in phase_two],
    )


# The (N, empty) pair and the six singleton pairs open every six-player run.
OPENING = "63 0 1 62 2 61 4 59 8 55 16 47 32 31"


def _masks(bitmasks: str) -> list[int]:
    return [int(value) for value in bitmasks.split()]


@pytest.mark.parametrize(
    ("options", "phase_one", "m_hat", "phase_two"),
    [
        (
            {},
            f"{OPENING} 33 30 18 45 12 51 10 53 35 28 50 13 9 54 20 43 48 15",
            [0.0, 0.0, 0.0, 8.0, 8.0, 0.0],
            (
                "44 19 41 22 7 56 14 49 26 37 21 42 52 11 38 25 "
                "27 36 60 3 57 6 39 24 23 40 58 5 29 34 46 17"
            ),
        ),
        (
            {"use_standard_method": True},
            f"{OPENING} 24 39 34 29 33 30 12 51 44 19 50 13 9 54 6 57 5 58 36 27 20 43",
            [0.0, 0.0, 0.0, 8.0, 6.0, 0.0],
            (
                "56 7 42 21 22 41 14 49 37 26 38 25 52 11 35 28 45 18 "
                "15 48 60 3 46 17 23 40 53 10"
            ),
        ),
        (
            {"initial_num_samples": 3},
            (
                f"{OPENING} 33 30 18 45 12 51 10 53 35 28 50 13 52 11 27 36 "
                "9 54 3 60 20 43 24 39 48 15"
            ),
            [0.0, 0.0, 0.0, 7.0, 5.0, 0.0],
            "49 14 7 56 38 25 42 21 44 19 22 41 26 37 46 17 29 34 57 6 58 5 23 40",
        ),
    ],
)
def test_seeded_two_phase_run_is_pinned(
    options: dict[str, float | bool],
    phase_one: str,
    m_hat: list[float],
    phase_two: str,
) -> None:
    """Six players, full budget (64 masks), seed 0: masks as bitmasks."""
    assert _run(6, 1.0, **options) == (_masks(phase_one), m_hat, _masks(phase_two))
    assert len(_masks(phase_one)) + len(_masks(phase_two)) == 64


def test_seeded_partial_budget_run_is_pinned() -> None:
    """Five players at fraction 0.75 (24 masks): phase one fills its grid with 22."""
    assert _run(5, 0.75) == (
        _masks("31 0 1 30 2 29 4 27 8 23 16 15 17 14 10 21 12 19 18 13 5 26"),
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [25, 6],
    )


@pytest.mark.parametrize(
    ("num_players", "fraction", "masks"),
    [
        (1, 1.0, [1, 0]),  # 2 masks: only (N, empty)
        (2, 1.0, [3, 0, 1, 2]),  # floor min(6, 4) = 4: one singleton pair
        # Floor 2n + 2 = 10: (N, empty) and the four singleton pairs, no walk.
        (4, 0.01, [15, 0, 1, 14, 2, 13, 4, 11, 8, 7]),
    ],
)
def test_phase_one_at_the_budget_floor(
    num_players: int, fraction: float, masks: list[int]
) -> None:
    sampled = NeymanEstimator().sample_masks(
        _players(num_players), fraction, False, seed=0
    )
    assert [present_to_mask_int(mask.present) for mask in sampled] == masks
