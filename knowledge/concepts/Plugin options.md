---
title: "Plugin options"
type: concept
domain: plugins
parent: "[[mllm-shap]]"
tags: [concept]
created: 2026-10-08
related-code:
  - "[[src/mllm_shap/plugins/options.py]]"
  - "[[src/mllm_shap/protocols/plugin.py]]"
  - "[[src/mllm_shap/plugins/context.py]]"
---

# Plugin options

Plugin options are per-plugin settings under `explanation.plugin_options`, keyed by `<group>/<name>` (for example `absence_policies/pad: {placeholder: "<unk>"}`) or by the bare plugin name when only one plugin of the run has it. A plugin class declares what it accepts in a `PluginOptions` model (`options_model`); `check_plugin_options` validates every entry before a run starts and raises `ConfigError` for unknown keys, wrong types or keys that name no plugin of the run, and the plugin reads its validated options from `PluginContext.options`.

Package: [[plugins]].
