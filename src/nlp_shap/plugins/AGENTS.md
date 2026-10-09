# Plugins agents

- New kind → class + `[project.entry-points."nlp_shap.<group>"]` in `pyproject.toml`
- Test via `PluginRegistry` + real behavior
- Scaffold: `uv run python .agents/skills/plugin/scripts/new_plugin.py --help`
- Any registry, context or builtin change → update `knowledge/architecture/plugins.md` (skill `notes`)

See `.cursor/rules/plugins.mdc` and skill `plugin`.
