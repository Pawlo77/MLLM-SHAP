---
name: review
description: >-
  Use when asked to review or audit a diff or a branch, or on /review.
---

# nlp-shap review

## Inputs

`git diff main...HEAD`, or a path the user names. Never a pull request on GitHub: Claude
reviews before the PR is opened, Cursor Bugbot reviews the PR. Read every changed file
fully; read the tests that cover it. Read `knowledge/handbook/Solutions/` first: a change
that repeats an approach a solution note records as failed, or ignores its reusable rule,
is a finding.

## Procedure

1. Run `make check`. A red check is the first finding.
2. Walk [checklist.md](checklist.md) top to bottom; every item is a yes/no.
3. For estimator, aggregator, masking or orchestrator changes run
   `uv run pytest tests/estimation/test_axioms.py tests/pipeline/test_pipeline_axioms.py -q`
   and add a property test if the change introduces a new code path.
4. For hot paths (`runtime/`, `estimation/`, `masking/`, backends) run `make bench`
   or `make bench-regression` (compares against `tests/benchmarks/baseline.json`).
5. For public API changes confirm `docs/` and `docs/release_notes.rst` changed.

## Output

One finding per line: `severity file:line — problem — fix`. Severities are the ones
`.cursor/BUGBOT.md` ("Severity") defines, quoted:

> - **Blocker:** leaks a secret, breaks CI or a release, or makes a result wrong (an estimator no
>   longer satisfies efficiency or no longer recovers a known-truth game, and its test was loosened).
> - **High:** a likely failure in a supported workflow (`make check`, `make docs`, explain from the
>   CLI or a notebook, publishing), or a wrong or misleading public name.
> - **Medium:** brittle or misleading; it will bite the next change.
> - **Low:** polish.

End with "Verdict: merge / fix first / reject"; any Blocker or High means "fix first".
No praise, no summary of what the diff does.
