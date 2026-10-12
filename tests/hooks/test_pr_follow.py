"""PR follow-through: ready only when green, up to date and without open comments.

@see [[scripts/pr_follow.py]]
"""

import importlib.util
import json
from types import ModuleType
from typing import Any

import pytest

from tests.hooks._hookrun import ROOT


@pytest.fixture(scope="module")
def follow() -> ModuleType:
    """Load scripts/pr_follow.py."""
    spec = importlib.util.spec_from_file_location(
        "pr_follow", ROOT / "scripts" / "pr_follow.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner(
    rollup: list[dict[str, Any]],
    *,
    state: str = "OPEN",
    merge: str = "CLEAN",
    decision: str = "",
    comments: list[dict[str, Any]] | None = None,
    author: str = "me",
    threads: list[dict[str, Any]] | None = None,
):
    view = {
        "number": 7,
        "url": "https://github.com/o/r/pull/7",
        "state": state,
        "author": {"login": author},
        "mergeStateStatus": merge,
        "reviewDecision": decision,
        "statusCheckRollup": rollup,
        "comments": comments or [],
    }
    graph = {
        "data": {
            "repository": {"pullRequest": {"reviewThreads": {"nodes": threads or []}}}
        }
    }

    def run(args: list[str]) -> str:
        return json.dumps(graph if args[0] == "api" else view)

    return run


GREEN = [
    {"name": "prek", "status": "COMPLETED", "conclusion": "SUCCESS"},
    {"name": "CodeQL", "status": "COMPLETED", "conclusion": "SKIPPED"},
    {"context": "codecov/patch", "state": "SUCCESS"},
]


def test_green_and_clean_is_ready(follow: ModuleType) -> None:
    report = follow.inspect("7", _runner(GREEN))
    assert report["ready"]
    assert report["todo"] == []


@pytest.mark.parametrize(
    ("rollup", "expected"),
    [
        (
            [{"name": "t", "status": "COMPLETED", "conclusion": "FAILURE"}],
            "failing check: t",
        ),
        ([{"name": "t", "status": "IN_PROGRESS"}], "pending check: t"),
        (
            [{"context": "codecov/patch", "state": "FAILURE"}],
            "failing check: codecov/patch",
        ),
        ([{"context": "ci", "state": "PENDING"}], "pending check: ci"),
    ],
)
def test_checks_block(
    follow: ModuleType, rollup: list[dict[str, Any]], expected: str
) -> None:
    report = follow.inspect("7", _runner(rollup))
    assert not report["ready"]
    assert expected in report["todo"]


@pytest.mark.parametrize("merge", ["BEHIND", "DIRTY"])
def test_stale_branch_blocks(follow: ModuleType, merge: str) -> None:
    assert not follow.inspect("7", _runner(GREEN, merge=merge))["ready"]


def test_unresolved_thread_and_changes_requested_block(follow: ModuleType) -> None:
    thread = {
        "isResolved": False,
        "isOutdated": False,
        "path": "a.py",
        "line": 3,
        "comments": {
            "nodes": [{"author": {"login": "copilot"}, "body": "Rename this\nx"}]
        },
    }
    resolved = {**thread, "isResolved": True}
    report = follow.inspect(
        "7", _runner(GREEN, decision="CHANGES_REQUESTED", threads=[thread, resolved])
    )
    assert "changes requested by a reviewer" in report["todo"]
    threads = [item for item in report["todo"] if item.startswith("unresolved thread")]
    assert threads == ["unresolved thread: a.py:3 copilot: Rename this"]


def test_comments_of_others_are_listed_and_keep_the_latest(follow: ModuleType) -> None:
    comments = [
        {"author": {"login": "codecov"}, "body": "old report"},
        {"author": {"login": "me"}, "body": "my own note"},
        {"author": {"login": "reviewer"}, "body": "please add a test"},
        {"author": {"login": "codecov"}, "body": "Coverage 81% (-2%)"},
    ]
    report = follow.inspect("7", _runner(GREEN, comments=comments))
    assert report["ready"]
    assert report["comments"] == [
        ("codecov", "Coverage 81% (-2%)"),
        ("reviewer", "please add a test"),
    ]
    assert "Coverage 81% (-2%)" in follow.render(report)


def test_main_exit_codes(
    follow: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert follow.main(["7"], _runner(GREEN)) == 0
    bad = [{"name": "t", "status": "COMPLETED", "conclusion": "FAILURE"}]
    assert follow.main(["7"], _runner(bad)) == 1

    def broken(_args: list[str]) -> str:
        raise RuntimeError

    assert follow.main(["7"], broken) == 2
    assert "NOT READY" in capsys.readouterr().out


def test_only_the_latest_run_of_a_check_counts(follow: ModuleType) -> None:
    cancelled = {"name": "title", "status": "COMPLETED", "conclusion": "CANCELLED"}
    passed = {"name": "title", "status": "COMPLETED", "conclusion": "SUCCESS"}
    early, late = (
        {"startedAt": "2026-10-12T10:07:24Z"},
        {"startedAt": "2026-10-12T10:07:36Z"},
    )
    recovered = [{**cancelled, **early}, {**passed, **late}]
    assert follow.inspect("7", _runner(recovered))["ready"]
    regressed = [{**passed, **early}, {**cancelled, **late}]
    assert follow.inspect("7", _runner(regressed))["todo"] == ["failing check: title"]


def test_review_bodies_are_listed(follow: ModuleType) -> None:
    view_runner = _runner(GREEN)

    def run(args: list[str]) -> str:
        data = json.loads(view_runner(args))
        if args[0] != "api":
            data["reviews"] = [
                {"author": {"login": "copilot"}, "state": "COMMENTED", "body": "quota"},
                {"author": {"login": "empty"}, "state": "APPROVED", "body": ""},
            ]
        return json.dumps(data)

    assert follow.inspect("7", run)["comments"] == [("copilot", "[COMMENTED] quota")]
