# Plugin

Add or register an nlp_shap plugin (skill `nlp-shap-plugin`):

1. Optional scaffold:
   `uv run python .cursor/skills/nlp-shap-plugin/scripts/new_plugin.py <group> <name> --dry-run`
2. Implement Protocol methods; entry point in `pyproject.toml`
3. Behavior test via `PluginRegistry`
4. Docs/notebook if user-facing
5. `make agent-check`
