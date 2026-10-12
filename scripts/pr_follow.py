"""Say whether a pull request is finished: green, up to date and without open comments.

Usage:
    python scripts/pr_follow.py [PR] [--json]

``PR`` defaults to the pull request of the current branch. Exit 0 means ready; exit 1
lists what is left (failing or pending checks, a branch behind or in conflict with its
base, requested changes, unresolved review threads); exit 2 means ``gh`` failed. The
latest comment of everyone but the PR's author (Codecov, Copilot, CodeQL, reviewers) is
always printed, because a coverage report or a remark is not a check and must be read,
not assumed. Stdlib only.
"""

import argparse
import json
import subprocess  # nosec B404 - fixed argv lists, no shell
import sys
import time
from collections.abc import Callable
from typing import Any

Runner = Callable[[list[str]], str]

FAILING = {"FAILURE", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"}
OK = {"SUCCESS", "NEUTRAL", "SKIPPED"}
THREADS = """query($o:String!,$r:String!,$n:Int!){repository(owner:$o,name:$r){
pullRequest(number:$n){reviewThreads(first:100){nodes{isResolved isOutdated path line
comments(first:1){nodes{author{login} body}}}}}}}"""
BODY_LIMIT = 700
"""Characters of a comment shown; enough for a Codecov summary table."""


def gh(args: list[str], attempts: int = 3) -> str:
    """Run ``gh``, retrying GitHub 5xx errors; return stdout or raise."""
    for attempt in range(1, attempts + 1):
        run = subprocess.run(  # nosec B603 B607 - fixed argv, gh from PATH
            ["gh", *args], capture_output=True, text=True, check=False
        )
        if not run.returncode:
            return run.stdout
        error = run.stderr.strip() or f"gh {' '.join(args)} failed"
        if "HTTP 5" not in error or attempt == attempts:
            raise RuntimeError(error)
        time.sleep(2 * attempt)
    raise RuntimeError("unreachable")


def classify_checks(rollup: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Return failing and pending check names, judging each name by its latest run."""
    latest: dict[str, dict[str, Any]] = {}
    for check in rollup:  # a rerun or a cancelled run leaves older entries of one name
        name = check.get("name") or check.get("context") or "?"
        stamp = check.get("startedAt") or check.get("completedAt") or ""
        if name not in latest or stamp >= (latest[name].get("startedAt") or ""):
            latest[name] = check
    failing, pending = [], []
    for name, check in latest.items():
        if "status" in check:  # CheckRun
            if check["status"] != "COMPLETED":
                pending.append(name)
            elif check.get("conclusion") in FAILING:
                failing.append(name)
            elif check.get("conclusion") not in OK:
                pending.append(name)
        elif check.get("state") in {"ERROR", "FAILURE"}:  # commit status
            failing.append(name)
        elif check.get("state") != "SUCCESS":
            pending.append(name)
    return failing, pending


def _review(review: dict[str, Any]) -> dict[str, Any]:
    """Return a review with a body as a comment labelled with its state."""
    login = review.get("author", {}).get("login", "")
    body = review.get("body", "")
    return {
        "author": {"login": login},
        "body": f"[{review.get('state')}] {body}" if body else "",
    }


def other_comments(
    comments: list[dict[str, Any]], pr_author: str
) -> list[tuple[str, str]]:
    """Return the latest ``(author, body)`` of each commenter except the PR's author.

    Apps such as Codecov have no ``[bot]`` suffix, so nobody is guessed to be a bot.
    """
    latest: dict[str, str] = {}
    for comment in comments:
        author = comment.get("author", {}).get("login", "")
        if author and author != pr_author and comment.get("body"):
            latest[author] = comment.get("body", "")
    return list(latest.items())


def inspect(number: str, run: Runner = gh) -> dict[str, Any]:
    """Collect the state of pull request ``number`` (``""`` for the current branch)."""
    fields = (
        "number,url,state,author,mergeStateStatus,reviewDecision,"
        "statusCheckRollup,comments,reviews"
    )
    view = json.loads(
        run(["pr", "view", *([number] if number else []), "--json", fields])
    )
    owner_repo = view["url"].split("/pull/")[0].removeprefix("https://github.com/")
    owner, repo = owner_repo.split("/")
    query = ["api", "graphql", "-f", f"query={THREADS}"]
    query += ["-F", f"o={owner}", "-F", f"r={repo}", "-F", f"n={view['number']}"]
    graph = json.loads(run(query))
    nodes = graph["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    failing, pending = classify_checks(view.get("statusCheckRollup") or [])
    threads = [
        f"{t['path']}:{t['line'] or '?'} "
        f"{t['comments']['nodes'][0]['author']['login']}: "
        f"{t['comments']['nodes'][0]['body'].splitlines()[0][:120]}"
        for t in nodes
        if not t["isResolved"] and t["comments"]["nodes"]
    ]
    todo = []
    if view["state"] != "OPEN":
        todo.append(f"pull request is {view['state'].lower()}")
    todo += [f"failing check: {name}" for name in failing]
    todo += [f"pending check: {name}" for name in pending]
    if view["mergeStateStatus"] == "BEHIND":
        todo.append("branch is behind its base: merge origin/main (no rebase)")
    if view["mergeStateStatus"] == "DIRTY":
        todo.append("merge conflicts with the base: merge origin/main and resolve")
    if view["reviewDecision"] == "CHANGES_REQUESTED":
        todo.append("changes requested by a reviewer")
    todo += [f"unresolved thread: {line}" for line in threads]
    return {
        "number": view["number"],
        "url": view["url"],
        "ready": not todo,
        "todo": todo,
        "comments": other_comments(
            [*(view.get("comments") or []), *map(_review, view.get("reviews") or [])],
            view.get("author", {}).get("login", ""),
        ),
    }


def render(report: dict[str, Any]) -> str:
    """Return the human-readable report."""
    lines = [f"{report['url']}: {'READY' if report['ready'] else 'NOT READY'}"]
    lines += [f"  - {item}" for item in report["todo"]]
    for author, body in report["comments"]:
        text = " ".join(body.split())[:BODY_LIMIT]
        lines.append(f"  comment by {author}, read it and act or say why not: {text}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, run: Runner = gh) -> int:
    """Print the report for the requested pull request."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pr", nargs="?", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = inspect(args.pr, run)
    except (RuntimeError, KeyError, ValueError) as error:
        print(f"pr_follow: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
