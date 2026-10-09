Configuration
=============

``nlp-shap`` uses one :class:`~nlp_shap.ExplainConfig` schema with three
top-level sections: ``backend`` (which model), ``generation`` (how it generates)
and ``explanation`` (what is explained and how). Every section rejects unknown
keys, so a typo fails validation instead of being ignored.

YAML skeleton
-------------

A complete configuration. Only ``backend.kind`` and ``backend.model_id`` are
required; every omitted key takes the default listed in the key reference below.
The prefix cache (``kv_cache``) only affects the ``transformers`` backend and is
off by default; ``max_players`` refuses, before any model call, runs whose budget
plans more than :math:`2^{\text{max\_players}}` coalitions (``budget.fraction``
times :math:`2^n` for :math:`n` players, so every full-table run on more than
``max_players`` players). ``budget.evaluations`` replaces ``budget.fraction``
with an absolute number of coalition evaluations; set one of the two, never
both:

.. code-block:: yaml

   backend:
     kind: lmstudio
     model_id: qwen2-500m-instruct

   generation:
     max_new_tokens: 128
     temperature: 0.0
     top_k: 1

   explanation:
     estimand: shapley
     estimator: neyman_cc
     value_fn: tfidf_cosine
     normalizer: identity
     players: tokens
     absence_policy: delete
     budget:
       fraction: 0.3
     include_minimal_masks: false
     neyman:
       use_standard_method: false
       initial_fraction: null
       initial_num_samples: null
     max_inflight: 2
     max_players: 24
     archive:
       path: ./runs/{run_id}
       flush_every: 50
       on_exists: error
     dedup:
       enabled: auto
     kv_cache:
       enabled: false
       max_entries: 8
     embedding_mode: static
     seed: 42
     plugin_options: {}

Load and save
-------------

:func:`~nlp_shap.explain_config_from_yaml` validates YAML text and
:func:`~nlp_shap.explain_config_to_yaml` writes every key back, defaults included:

.. code-block:: python

   from pathlib import Path

   from nlp_shap import ExplainConfig, explain_config_from_yaml, explain_config_to_yaml

   Path("config.yaml").write_text(
       "backend:\n"
       "  kind: mock\n"
       "  model_id: stub\n"
       "explanation:\n"
       "  estimator: exact\n",
       encoding="utf-8",
   )
   config = explain_config_from_yaml(Path("config.yaml").read_text(encoding="utf-8"))
   assert isinstance(config, ExplainConfig)

   text = explain_config_to_yaml(config)
   Path("config.roundtrip.yaml").write_text(text, encoding="utf-8")
   print("\n".join(text.splitlines()[:8]))

.. guide-result:: config_roundtrip

In Python, build the same object with ``ExplainConfig.model_validate({...})``;
configs are frozen, so derive variants with ``model_copy(update=...)``.

Key reference
-------------

Generated from the pydantic models, so it always matches the installed version.

``backend``
~~~~~~~~~~~

.. config-table:: nlp_shap.pipeline.config.BackendConfig
   :prefix: backend

``generation``
~~~~~~~~~~~~~~

.. config-table:: nlp_shap.pipeline.config.GenerationConfig
   :prefix: generation

``explanation``
~~~~~~~~~~~~~~~

.. config-table:: nlp_shap.pipeline.config.ExplanationConfig
   :prefix: explanation

Plugin options
--------------

``explanation.plugin_options`` passes settings to one plugin of the run. Key them
by ``<group>/<name>`` (groups: ``estimators``, ``estimands``, ``value_fns``,
``normalizers``, ``backends``, ``partitions``, ``absence_policies``,
``renderers``) or by the bare plugin name when no other plugin of the run shares
it. The registry hands them to the plugin as
:attr:`PluginContext.options <nlp_shap.plugins.context.PluginContext.options>`;
see :doc:`extending` for a plugin that reads them. Built-in plugins with options
are ``absence_policies/pad`` (``placeholder``), ``absence_policies/neutral``
(``fill_char``) and ``normalizers/power_shift`` (``power``); the others accept
none. Before a run starts, a key naming no plugin of the run, an unknown option
or a value of the wrong type raises :exc:`~nlp_shap.errors.ConfigError` naming
the group, plugin and key.

.. code-block:: yaml

   explanation:
     value_fn: my_metric
     plugin_options:
       value_fns/my_metric:
         threshold: 0.5

Archive path and re-runs
------------------------

``explanation.archive.path`` is a template; ``{run_id}`` expands to the first 16
characters of the snapshot id plus a 12-character digest of the configuration
(archive settings excluded) and the backend model id. Changing the estimator,
policy, value function, seed or model therefore writes to a new directory.

Re-running the same configuration on the same snapshot targets the same
directory. With ``on_exists: error`` (the default) the run raises
:exc:`~nlp_shap.errors.ArchiveExistsError` instead of appending to the old
archive; ``on_exists: overwrite`` replaces it. Set ``path: ""`` to skip
archiving entirely.

Example configurations
----------------------

The repository ships one configuration per local backend in ``config/``
(`dev_api.yaml
<https://github.com/Pawlo77/nlp-shap/blob/main/config/dev_api.yaml>`_,
`dev_lmstudio.yaml
<https://github.com/Pawlo77/nlp-shap/blob/main/config/dev_lmstudio.yaml>`_,
`dev_transformers.yaml
<https://github.com/Pawlo77/nlp-shap/blob/main/config/dev_transformers.yaml>`_);
:doc:`backends` explains each backend's keys.

Further reading
---------------

- :doc:`pipeline` — what each section controls during a run
- :doc:`choosing_an_estimator` — ``estimator`` and ``budget.fraction``
- :doc:`masking` — ``players`` and ``absence_policy``
- :doc:`runtime` — ``archive``, ``dedup`` and ``max_inflight``
- :doc:`migrating_from_mllm_shap` — mapping from MLLM-Shap settings
