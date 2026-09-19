"""Tests for sessionStart next-phase helper."""

import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / ".cursor" / "hooks" / "next_phase.py"


@pytest.fixture(scope="module")
def next_phase():
    """Load next_phase.py without installing it as a package."""
    spec = importlib.util.spec_from_file_location("next_phase", _HOOK)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("version", "needle"),
    [
        ("0.1.16", "phase 15"),
        ("0.2.0", "phase 16"),
        ("0.2.5", "phase 21"),
        ("0.2.6", "complete"),
        ("1.0.0", "complete"),
    ],
)
def test_next_phase_hint(next_phase, version: str, needle: str) -> None:
    """Version maps to the next unfinished audio phase (or complete)."""
    assert needle in next_phase.next_phase_hint(version).lower()
