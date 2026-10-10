<div align="center">
  <h1>Examples</h1>
  <p><strong>End-to-end Jupyter notebooks for nlp-shap explainability workflows.</strong></p>
</div>

## Setup

Install the package, then open a notebook with Jupyter or VS Code.

```bash
pip install nlp-shap                      # from PyPI
git clone https://github.com/Pawlo77/nlp-shap && cd nlp-shap && make install   # from a clone
```

## Run

From a clone, `make notebooks` re-executes every notebook in place and stores its outputs. To work interactively, run `make kernel` (kernel `nlp-shap (uv)`) or select `.venv` as the kernel in VS Code. A notebook that needs an optional extra says so in its first cell; install it with `uv sync --extra <name>`.

## Notebooks

What each notebook demonstrates, and the extras it needs: [docs gallery](https://pawlo77.github.io/nlp-shap/examples.html) (source `docs/examples.rst`).

- `explain_pipeline.ipynb`
- `archive_reanalyze.ipynb`
- `cli.ipynb`
- `plugins_extending.ipynb`
- `estimands_toy_game.ipynb`
- `masking_views.ipynb`
- `runtime_core.ipynb`
- `exact_estimation.ipynb`
- `estimator_comparison.ipynb`
- `attribution_viz.ipynb`
- `text_explain_e2e.ipynb`
- `audio_alignment.ipynb`
