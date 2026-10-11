"""Probe Shapley efficiency and additive recovery through the public pipeline.

Runs every built-in estimator on 1 to 6 tokens with the echo backend and the
count value function from ``tests/pipeline/test_pipeline_axioms.py``. In that
game a token scores 1 when it is in the value function's vocabulary, so the
exact attribution of each token is known. Prints one PASS/FAIL line per check
and exits 1 when any check fails.

Usage::

    uv run python .agents/skills/bugspot/scripts/probe_axioms.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mllm_shap import ExplainRunner  # noqa: E402
from tests.pipeline.test_pipeline_axioms import (  # noqa: E402
    CountValue,
    _config,
    _registry,
    _snapshot,
)

ESTIMATORS: tuple[str, ...] = ("exact", "mc", "complementary", "neyman_cc")
"""Built-in estimator names probed, in report order."""

TOKENS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
"""Prompt tokens; prefixes of length 1 to 6 are probed."""

TOLERANCE = 1e-9
"""Absolute tolerance for both checks (the game is exact and deterministic)."""


def probe(estimator: str, n_tokens: int) -> list[tuple[str, bool, str]]:
    """Run one estimator on ``n_tokens`` tokens and evaluate both axioms.

    Args:
        estimator: Built-in estimator name.
        n_tokens: Number of prompt tokens (players).

    Returns:
        ``(check name, passed, detail)`` per check; one ``error`` entry when
        the run raises.
    """
    words = TOKENS[:n_tokens]
    try:
        output = ExplainRunner(_config(estimator), registry=_registry()).explain_sync(
            _snapshot(" ".join(words))
        )
    except Exception as exc:
        return [("run", False, f"{type(exc).__name__}: {exc}")]
    result = output.result
    values = [float(v) for v in result.values]
    gap = sum(values) - (result.grand_value - result.base_value)
    expected = [1.0 if w in CountValue.vocabulary else 0.0 for w in words]
    worst = max((abs(v - e) for v, e in zip(values, expected, strict=False)), default=0)
    return [
        ("efficiency", abs(gap) <= TOLERANCE, f"sum - (grand - base) = {gap:+.3g}"),
        (
            "additive",
            len(values) == len(expected) and worst <= TOLERANCE,
            f"values={[round(v, 6) for v in values]} expected={expected}",
        ),
    ]


def main() -> int:
    """Run every probe, print the report, and return the exit code."""
    failures = 0
    for estimator in ESTIMATORS:
        for n_tokens in range(1, len(TOKENS) + 1):
            for check, passed, detail in probe(estimator, n_tokens):
                failures += not passed
                status = "PASS" if passed else "FAIL"
                print(f"{status} {estimator:<13} n={n_tokens} {check:<10} {detail}")
    print(f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
