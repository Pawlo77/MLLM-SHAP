"""The review and bug-hunting skills carry trigger evals (skill-creator format).

@see [[scripts/sync_agents.py]]
"""

from tests.hooks._hookrun import ROOT


def test_trigger_evals_exist_for_review_and_bugspot() -> None:
    skills = ROOT / ".agents" / "skills"
    names = {path.parents[1].name for path in skills.glob("*/evals/trigger-evals.json")}
    assert {"review", "bugspot"} <= names
