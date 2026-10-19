Generative backends
===================

``mllm-shap`` sends the base prompt and every coalition prompt to a **generative
backend**. Each backend implements
:class:`~mllm_shap.protocols.backend.GenerativeBackend` and is selected with
``backend.kind`` in :class:`~mllm_shap.ExplainConfig`. The rest of the pipeline is
the same for every backend.

Built-in backends
-----------------

.. list-table::
   :header-rows: 1
   :widths: 16 18 66

   * - ``kind``
     - Extra
     - Use when
   * - ``mock``
     - *(core)*
     - Tests, CI and offline demos; no model involved
   * - ``mock_audio``
     - *(core)*
     - Audio runs in tests and demos (:doc:`audio_backends`)
   * - ``api``
     - ``[api]``
     - Remote or local OpenAI-compatible HTTP servers (OpenAI, vLLM, LM Studio's
       server mode, proxies)
   * - ``lmstudio``
     - ``[lmstudio]``
     - Local models through the LM Studio SDK, with automatic download
   * - ``transformers``
     - ``[transformers]``
     - Hugging Face language models in-process, with prefix-cache reuse;
       audio language models hear audio parts (:doc:`audio_backends`)

Settings shared by all backends are the top-level ``backend`` keys (see the
reference in :doc:`config`); settings that only one backend understands go under
``backend.options`` and are validated by that backend.

Speakers and token logprobs
---------------------------

Chat templates have no speaker slot, so every backend sends a message with a
``speaker`` as ``"<speaker>: <text>"`` content under its role (``user``,
``assistant`` or ``system``). A message emptied by the ``delete`` policy is sent
empty, without the name.

The ``self_logprob`` value function (:doc:`value_functions`) needs the log probability
of each generated token:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - ``kind``
     - Token logprobs
   * - ``transformers``
     - Always: log-softmax of the raw logits at each generated token, the
       end-of-sequence token included, with or without the KV cache
   * - ``api``
     - Requests ``"logprobs": true`` and reads
       ``choices[0].logprobs.content[*].logprob`` (OpenAI, LM Studio's ``/v1``);
       ``None`` when the server omits them
   * - ``lmstudio``
     - None: the SDK's prediction result carries no logprobs; use ``api`` with
       LM Studio's server instead
   * - ``mock``, ``mock_audio``
     - None

The teacher-forced value functions (``tf_logprob``, ``tf_jsd``, ``contrastive``) need
:class:`~mllm_shap.protocols.backend.SupportsTeacherForcing`: ``transformers`` (text
prompts, one forward pass per continuation) and ``mock`` (hash-derived
distributions) have it; ``api`` and ``lmstudio`` do not, because chat servers do
not return the logprobs of a prompt, and a run with them fails with
:exc:`~mllm_shap.errors.ConfigError` before generating.

Mock
----

:class:`~mllm_shap.backends.mock.MockBackend` returns a hash of the rendered prompt
and the model id as its "generation". It is deterministic and fast, so every guide
and the test suite use it. The text has no meaning: attributions computed with it
test the plumbing, not a model.

.. code-block:: yaml

   backend:
     kind: mock
     model_id: stub

``mock_audio`` hears audio parts; see :doc:`audio_backends`.

HTTP API
--------

:class:`~mllm_shap.backends.api.ApiBackend` posts chat-completions requests to
``{api_host}/chat/completions``. At ``temperature: 0.0`` it caches responses keyed
by the request body, so identical coalition prompts cost one HTTP call.

.. code-block:: yaml

   backend:
     kind: api
     model_id: gpt-4o-mini
     api_host: https://api.openai.com/v1
     timeout_s: 120
     max_retries: 3

   generation:
     temperature: 0.0

``timeout_s`` bounds each request. Timeouts, refused connections and HTTP 408,
429, 500, 502, 503 and 504 responses are retried up to ``max_retries`` times with
exponential backoff (0.5 s doubling, at most 8 s, or the server's
``Retry-After``). Any other error status, a response that is not JSON, or running
out of retries raises :exc:`~mllm_shap.errors.BackendUnavailableError`. The
backend keeps at most 1024 cached responses and opens at most
``explanation.max_inflight`` connections. Contract tests in CI mock the HTTP
transport; point ``api_host`` at a live server only for manual runs.

LM Studio
---------

:class:`~mllm_shap.backends.lmstudio.LmStudioBackend` talks to a running
`LM Studio <https://lmstudio.ai>`_ through its Python SDK. It looks for a
downloaded model whose key matches ``model_id`` and, when none is found and
``auto_download`` is on, searches the LM Studio hub (``hub_search`` or
``model_id``) and downloads the ``quantization`` variant. LM Studio's MLX runtime
does not accept concurrent requests on one model, so ``serialize_generate``
(default on) runs one ``generate`` at a time whatever ``max_inflight`` says.

``model_id`` is matched against downloaded models exactly (after normalization)
first, then as the prefix of the last ``/`` segment, then as a substring; more
than one match at a stage raises :exc:`~mllm_shap.errors.BackendUnavailableError`
listing the candidates. Only connection failures (``OSError`` and the SDK's
websocket or connection errors) are retried, once; ``ValueError`` and
``TypeError`` propagate unchanged.

.. code-block:: yaml

   backend:
     kind: lmstudio
     model_id: qwen2-500m-instruct
     options:
       quantization: Q4_K_M
       auto_download: true
       serialize_generate: true

.. config-table:: mllm_shap.backends.lmstudio.options.LmStudioOptions
   :prefix: backend.options

Integration tests that need a running LM Studio with ``qwen2-500m-instruct`` are
marked ``lms`` and excluded from ``make check``:

.. code-block:: bash

   uv run pytest -m lms tests/

Hugging Face transformers
-------------------------

:class:`~mllm_shap.backends.transformers.TransformersTextBackend` loads
``model_id`` on the first call (``AutoModelForCausalLM``, or a processor and the
checkpoint's class for audio models, :doc:`audio_backends`), on CUDA or Apple MPS
when available, otherwise on CPU. ``backend.options.dtype`` picks the weight
dtype: ``auto`` (default) uses ``bfloat16`` on accelerators that support it,
``float16`` on other accelerators and ``float32`` on CPU; ``float32``,
``float16`` and ``bfloat16`` force one. ``backend.options.revision`` pins the Hub
revision (a commit hash makes runs reproducible). It needs ``transformers>=4.56``.
With ``explanation.kv_cache.enabled`` (off by default) it caches the key/value
states of up to ``kv_cache.max_entries`` prompt prefixes (all prompt tokens but the
last) and reuses the longest match for each coalition prompt. The reuse is counted
in ``metrics.kv_cache_hits`` and ``metrics.kv_cache_reused_tokens``. A hit copies
the cached state, crops it to the shared prefix (Hugging Face
``DynamicCache.crop``), computes the missing prefix tokens and caches the completed
prefix, so coalitions reuse a shared prefix in any order. In ``float32`` greedy outputs
are the same with the cache on and off. In ``float16`` and ``bfloat16`` the cached
prefix runs in separate forward passes that round differently, so a near-tied
greedy token can flip and values can move slightly; the backend logs one warning,
and ``dtype: float32`` gives identical results. Static and
contextual embeddings of the output are computed only for value functions that
need them (``embedding_cosine`` with ``encoder: model_states``). The tiny random model below loads in seconds and is what the
``gpu``-marked tests use:

.. code-block:: yaml

   backend:
     kind: transformers
     model_id: hf-internal-testing/tiny-random-gpt2
     options:
       dtype: auto

   generation:
     max_new_tokens: 8

   explanation:
     max_inflight: 1
     kv_cache:
       enabled: true

.. code-block:: bash

   uv run pytest -m gpu tests/

Environment variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Effect
   * - ``MLLM_SHAP_API_KEY``
     - Bearer token the ``api`` backend sends to any host; it takes precedence
       over ``OPENAI_API_KEY``
   * - ``OPENAI_API_KEY``
     - Fallback bearer token, sent by the ``api`` backend only when
       ``MLLM_SHAP_API_KEY`` is not set and the host of ``backend.api_host`` is
       ``api.openai.com``, so an OpenAI key never reaches a local or third-party
       server
   * - ``MLLM_SHAP_LOGGING_CONFIG``
     - Path to a TOML file whose ``[tool.logging]`` section configures the
       ``mllm_shap`` loggers

Example configurations
----------------------

The repository keeps one runnable configuration per backend in ``config/``:
`dev_api.yaml
<https://github.com/Pawlo77/MLLM-SHAP/blob/main/config/dev_api.yaml>`_,
`dev_lmstudio.yaml
<https://github.com/Pawlo77/MLLM-SHAP/blob/main/config/dev_lmstudio.yaml>`_ and
`dev_transformers.yaml
<https://github.com/Pawlo77/MLLM-SHAP/blob/main/config/dev_transformers.yaml>`_.
Each file starts with a comment saying what it needs and how to run it.

Install the extra of the backend you use:

.. code-block:: bash

   pip install "mllm-shap[api]"
   pip install "mllm-shap[lmstudio]"
   pip install "mllm-shap[transformers]"

To add a backend of your own, see :doc:`extending`.

Further reading
---------------

- Configuration schema: :doc:`config`
- What a run does with the backend: :doc:`pipeline`
- API module reference: :doc:`../api/backends`
- ``examples/text_explain_e2e.ipynb`` (embedded on :doc:`../examples`) — one
  prompt on the LM Studio, transformers and HTTP API backends
