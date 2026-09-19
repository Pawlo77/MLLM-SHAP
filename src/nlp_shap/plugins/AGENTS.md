# Plugins agents

- New kind → class + `[project.entry-points."nlp_shap.<group>"]` in `pyproject.toml`
- Test via `PluginRegistry` + real behavior
- Scaffold: `uv run python .cursor/skills/nlp-shap-plugin/scripts/new_plugin.py --help`

See `.cursor/rules/plugins.mdc` and skill `nlp-shap-plugin`.
