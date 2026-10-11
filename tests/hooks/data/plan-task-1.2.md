### Task 1.2: Stratified marginal aggregation for partial samples (fixes C5)

**Files:**
- Modify: `src/mllm_shap/estimation/estimands/_weights.py:97-158`
- Modify: `tests/estimation/test_axioms.py` (append)

**Interfaces:**
- Produces: `aggregate_from_marginals(masks, payoffs, weight_fn)` keeps its signature but now returns the stratified mean estimator described below. `ShapleyAggregator.aggregate` and `BanzhafAggregator.aggregate` keep signatures.

Design: for player i and coalition size k (i ∉ S, |S| = k), the Shapley value is `(1/n) · E_k[v(S∪{i}) − v(S)]` summed over k = 0..n−1; the Banzhaf value is the plain mean of all observed marginals. With a partial sample, estimate each stratum mean from the observed pairs and skip strata with no observation, rescaling by `n / observed_strata` so the estimate stays on the scale of the full sum. This is consistent (converges to the exact value as coverage grows) and exact for additive games at any coverage.

- [ ] **Step 1: Write the failing test**

Append to `tests/estimation/test_axioms.py`:

```python
from mllm_shap.estimation.estimands._weights import (
    aggregate_from_marginals,
    banzhaf_weight,
    shapley_weight,
)


@pytest.mark.parametrize("num_players", [3, 5, 7])
def test_partial_sample_additive_game_is_unbiased(num_players: int) -> None:
    coefficients = [float(index + 1) for index in range(num_players)]
    masks = full_table(num_players)
    payoffs = additive_payoffs(masks, coefficients)
    # Keep every other coalition plus the empty and grand coalition.
    keep = [i for i in range(len(masks)) if i % 2 == 0 or i in (0, len(masks) - 1)]
    sub_masks = [masks[i] for i in keep]
    sub_payoffs = [payoffs[i] for i in keep]
    shapley = aggregate_from_marginals(sub_masks, sub_payoffs, shapley_weight)
    banzhaf = aggregate_from_marginals(sub_masks, sub_payoffs, banzhaf_weight)
    assert shapley == pytest.approx(coefficients, abs=1e-9)
    assert banzhaf == pytest.approx(coefficients, abs=1e-9)


def test_partial_sample_scale_matches_full_table() -> None:
    masks = full_table(4)
    payoffs = [float(sum(m) ** 2) for m in masks]
    full = aggregate_from_marginals(masks, payoffs, shapley_weight)
    half = aggregate_from_marginals(
        masks[::2] + [masks[-1]], payoffs[::2] + [payoffs[-1]], shapley_weight
    )
    assert sum(half) == pytest.approx(sum(full), rel=0.35)
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest tests/estimation/test_axioms.py -k partial -q
```
Expected: FAIL, `test_partial_sample_additive_game_is_unbiased` values below coefficients (biased toward zero).

- [ ] **Step 3: Replace `aggregate_from_marginals`**

Replace lines 97–158 of `src/mllm_shap/estimation/estimands/_weights.py` with:

```python
def aggregate_from_marginals(
    masks: Sequence[Sequence[bool]],
    payoffs: Sequence[float],
    weight_fn: Callable[[int, int], float],
) -> list[float]:
    """Estimate per-player values from observed marginal contributions.

    Each observed pair ``(S, S ∪ {i})`` with ``i ∉ S`` contributes the marginal
    ``v(S ∪ {i}) − v(S)`` to the stratum ``|S|``. Under Shapley weights every
    stratum carries weight ``1/n`` and the estimate is the mean marginal inside
    the stratum; strata without observations are skipped and the sum is rescaled
    by ``n / observed_strata``. Under uniform (Banzhaf) weights the estimate is
    the plain mean of all observed marginals. On a complete table both reduce to
    the exact value.

    Args:
        masks: Coalition membership rows with shape ``(m, n)``.
        payoffs: Characteristic-function samples ``v(S)`` aligned with masks.
        weight_fn: Coalition weight callback receiving ``(coalition_size, n)``.

    Returns:
        Per-player attribution vector with length ``n``.

    Raises:
        ValueError: If masks are empty, ragged, or misaligned with payoffs.
    """
    if not masks:
        raise ValueError("masks must not be empty")
    if len(masks) != len(payoffs):
        raise ValueError("masks and payoffs must have the same length")

    num_players = len(masks[0])
    if num_players == 0:
        raise ValueError("masks must include at least one player")
    if any(len(mask) != num_players for mask in masks):
        raise ValueError("all masks must have the same number of players")

    payoff_lookup: dict[int, float] = {}
    for mask, payoff in zip(masks, payoffs, strict=True):
        payoff_lookup[_mask_as_int(mask)] = payoff

    uniform = weight_fn is banzhaf_weight
    values = [0.0] * num_players
    for player_index in range(num_players):
        player_bit = 1 << player_index
        stratum_sum = np.zeros(num_players, dtype=np.float64)
        stratum_count = np.zeros(num_players, dtype=np.int64)
        for mask_int, payoff in payoff_lookup.items():
            if mask_int & player_bit:
                continue
            extended_payoff = payoff_lookup.get(mask_int | player_bit)
            if extended_payoff is None:
                continue
            size = mask_int.bit_count()
            stratum_sum[size] += extended_payoff - payoff
            stratum_count[size] += 1

        observed = stratum_count > 0
        if not observed.any():
            continue
        if uniform:
            values[player_index] = float(stratum_sum.sum() / stratum_count.sum())
            continue
        stratum_mean = stratum_sum[observed] / stratum_count[observed]
        sizes = np.flatnonzero(observed)
        weights = np.array(
            [
                weight_fn(int(size), num_players)
                * math.comb(num_players - 1, int(size))
                for size in sizes
            ],
            dtype=np.float64,
        )
        # weight_fn(k, n) * C(n-1, k) == 1/n for Shapley; rescale skipped strata.
        values[player_index] = float(
            (weights * stratum_mean).sum() * num_players / len(sizes)
        )

    return values
```

Also delete the now-unused `_factorial_table` helper (lines 9–14) and its import usage.

- [ ] **Step 4: Run all estimation tests**

```bash
uv run pytest tests/estimation -q
```
Expected: all pass. If `tests/estimation/test_monte_carlo.py::test_mc_converges_toward_exact_on_additive_game` fails, it is because it builds its reference without the grand coalition; Task 1.3 fixes it.

- [ ] **Step 5: Commit**

```bash
git add src/mllm_shap/estimation/estimands/_weights.py tests/estimation/test_axioms.py
git commit -m "fix: make partial-sample marginal aggregation a stratified mean"
```
