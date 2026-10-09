# Review checklist

Every item is a yes/no question about the diff. A "no" is a finding; cite `file:line`.

## Correctness and axioms

- [ ] Efficiency is preserved: for Shapley estimands `sum(values) == grand_value - base_value` within the test tolerance.
- [ ] The empty and the grand coalition are still evaluated by every estimator path.
- [ ] Complementary-contribution pairs are still complementary (`S` and `N \ S`, including `(N, ∅)`).
- [ ] Results are deterministic for a fixed seed; every random draw takes the seeded generator.
- [ ] New code does not materialise all `2^n` coalitions outside the exact estimator.
- [ ] Scheduler and backend errors propagate to the caller; nothing is swallowed or left as an unretrieved task.

## Contracts and persistence

- [ ] The archive schema version is bumped when archive columns change.
- [ ] Manifest fields are added when the run configuration grows.
- [ ] Protocol signatures are unchanged, or every implementer and test double is updated.
- [ ] Imports inside `src/nlp_shap/` are relative; tests import `from nlp_shap...`.
- [ ] Public classes, functions and constants have Google docstrings.
- [ ] No `dict[str, Any]` where a `TypedDict` already exists.
- [ ] No `getattr`/`hasattr` capability probing; capability protocols are used instead.

## Tests and performance

- [ ] Tests assert values (known-truth games, exact numbers), not only shapes or types.
- [ ] No assertion was deleted or loosened (wider tolerance, smaller sample) without a stated reason.
- [ ] A hot-path change (`runtime/`, `estimation/`, `masking/`, backends) has a benchmark.

## Docs, notes and hygiene

- [ ] Public API or behaviour changes update `docs/` and `docs/release_notes.rst`.
- [ ] Notebooks were re-executed with outputs stored (`make notebooks`) when they changed, when a `src/` change touched a public workflow, and for a release PR.
- [ ] User-facing docs contain no phase numbers or rewrite roadmap.
- [ ] Every changed `src/nlp_shap/<pkg>/` has its `knowledge/architecture/<pkg>.md` updated; design choices have a section in their `knowledge/decisions/` log; fixed or found bugs update their `knowledge/issues/` note.
- [ ] `make vault-check` passes; `knowledge/generated/` changes come from `make knowledge`, not hand edits; new modules carry their traceability tags (`@domain`, `@see`, `@tested-by`).
- [ ] No new runtime dependency, or the dependency is justified in `pyproject.toml`.
- [ ] No secrets, tokens, private hostnames or personal data in code, notes or memory files.
