# Check

Run the nlp-shap validate loop (skill `nlp-shap-check`):

1. `make agent-check` (filtered pytest + prek-all)
2. If `docs/` or public API changed → `make docs`
3. If `examples/*.ipynb` changed → `make notebooks`
4. Report pass/fail; do not commit unless asked
