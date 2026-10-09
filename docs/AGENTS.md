# Docs agents

Rules: `.cursor/rules/docs.mdc`. Only what is specific to this directory is listed here.

- A ``.. code-block:: python`` followed by ``.. guide-result:: <id>`` is executed by
  `docs/scripts/render_guide_outputs.py` (page namespace, temporary working directory); never
  keep a second copy of a snippet.
- `make docs` is strict (`-W --keep-going -n`); fix references instead of ignoring them, and
  justify every `nitpick_ignore` entry in `conf.py`.
- `uv run pytest tests/docs -q` and `uv run python docs/scripts/check_docs.py` guard snippets,
  links, toctrees and API coverage.
