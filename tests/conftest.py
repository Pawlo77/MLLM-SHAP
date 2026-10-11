"""Shared fixtures for the mllm_shap test suite.

@see [[tests/AGENTS.md]]
"""

from dataclasses import dataclass, field

import pytest

from mllm_shap.domain.conversation import ConversationSnapshot, Message, Turn
from mllm_shap.domain.enums import Role
from mllm_shap.domain.estimands import Estimand
from mllm_shap.pipeline.manifest import RunManifest


@pytest.fixture
def sample_snapshot() -> ConversationSnapshot:
    """Return a short two-message conversation snapshot."""
    turn = Turn(
        messages=(
            Message(role=Role.USER, text="Who are you?"),
            Message(role=Role.ASSISTANT, text="I am helpful."),
        )
    )
    return ConversationSnapshot.from_turns((turn,))


@pytest.fixture
def run_manifest() -> RunManifest:
    """Return a minimal manifest for opening run archives."""
    return RunManifest(estimand=Estimand.SHAPLEY, run_id="test-run")


@dataclass
class RecordingProgress:
    """Capture progress callbacks for assertions."""

    planned: list[int] = field(default_factory=list)
    finished: list[tuple[int, int]] = field(default_factory=list)

    def on_coalitions_planned(self, total: int) -> None:
        self.planned.append(total)

    def on_coalition_finished(self, done: int, total: int) -> None:
        self.finished.append((done, total))


@pytest.fixture
def progress() -> RecordingProgress:
    """Return a fresh coalition progress recorder."""
    return RecordingProgress()
