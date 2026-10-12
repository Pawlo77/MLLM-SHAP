# CI: MLLM-SHAP jobs

`make docs` covers Deploy Docs; `make notebooks` covers the notebook job. A matrix failure on
one Python version only: `UV_PYTHON=3.14 uv run pytest ...`.
