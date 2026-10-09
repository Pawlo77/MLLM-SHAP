Command line
============

Installing ``nlp-shap`` adds an ``nlp-shap`` command for explaining prompts
without writing Python. It reads the same YAML configuration as
:class:`~nlp_shap.ExplainRunner` (see :doc:`config`) and a snapshot in JSON:

.. code-block:: bash

   nlp-shap config --example > config.yaml     # commented YAML skeleton
   nlp-shap config --check config.yaml         # validate a configuration
   nlp-shap explain --config config.yaml --snapshot snapshot.json --out result.json
   nlp-shap explain --config config.yaml --snapshot snapshot.json \
       --render token_bar --png result.png
   nlp-shap reanalyze --config config.yaml --archive runs/<run_id>
   nlp-shap plugins                            # registered plugins per group

Add ``--log-level info`` (``debug``, ``info``, ``warning`` or ``error``, any case)
before the command to see the pipeline's stage events on stderr (see
:doc:`runtime`). Exit codes: ``0`` success, ``2`` an invalid argument,
configuration, snapshot or archive, or an input file that cannot be read, ``3``
the backend is unavailable (a missing extra, an unreachable server or a refused
connection). Any other error is a bug and propagates with its traceback.

The snippets below call :func:`nlp_shap.cli.main` with the same arguments, which
is what the command runs. They share one working directory.

Snapshot JSON
-------------

A snapshot file lists turns, each with ordered messages; roles are ``user``,
``assistant`` and ``system``; each message is one text, with an optional
single-line ``"speaker"`` name for conversations where several people share a
role. The snapshot id is derived from the content (the speaker included), so the
same file always maps to the same run; a snapshot built with its own id carries
it as a top-level ``"snapshot_id"`` key.
:meth:`~nlp_shap.ConversationSnapshot.to_json` and
:meth:`~nlp_shap.ConversationSnapshot.from_json` write and read this format:

.. code-block:: python

   from pathlib import Path

   from nlp_shap import ConversationSnapshot, Message, Role, Turn

   snapshot = ConversationSnapshot.from_turns((
       Turn(
           messages=(
               Message(role=Role.USER, text="Why is the sky blue", speaker="ana"),
           )
       ),
   ))
   Path("snapshot.json").write_text(snapshot.to_json(), encoding="utf-8")
   print(Path("snapshot.json").read_text(encoding="utf-8"))
   restored = ConversationSnapshot.from_json(Path("snapshot.json").read_text())
   print("same snapshot:", restored == snapshot)

.. guide-result:: cli_snapshot

Explain
-------

``nlp-shap config --example`` prints a skeleton that runs as is with the mock
backend; change ``backend`` to point at a real model. ``explain`` writes the
result JSON to ``--out`` (or standard output). Its keys come in a fixed order:
``values``, ``labels`` (one entry per player with its text, character span,
speaker and display label),
``base_value``, ``grand_value``, ``estimand``, ``metrics``, ``manifest`` and
``perf``:

.. code-block:: python

   import json

   from nlp_shap.cli import EXAMPLE_CONFIG, main

   Path("config.yaml").write_text(EXAMPLE_CONFIG, encoding="utf-8")
   code = main([
       "explain",
       "--config", "config.yaml",
       "--snapshot", "snapshot.json",
       "--out", "result.json",
   ])
   result = json.loads(Path("result.json").read_text(encoding="utf-8"))
   print("exit code:", code)
   print("keys:", list(result))
   for label, value in zip(result["labels"], result["values"], strict=True):
       print(f"{label['text']:>5} {value:+.4f}")
   print("sum == grand - base:", round(sum(result["values"]), 9)
         == round(result["grand_value"] - result["base_value"], 9))

.. guide-result:: cli_explain

``--render token_text`` or ``--render token_bar`` together with ``--png`` also
saves the chart of :doc:`visualization` (needs the ``viz`` extra).

Reanalyze
---------

``reanalyze`` rescores an archived run without calling the model; edit
``value_fn`` or ``normalizer`` in the configuration first. The archive path comes
from ``explanation.archive.path`` (``./runs/{run_id}`` in the skeleton). A
reanalyzed result has no snapshot, so its labels carry the player ids only:

.. code-block:: python

   run_dir = Path("runs") / result["manifest"]["run_id"]
   rescored_config = EXAMPLE_CONFIG.replace(
       "normalizer: identity", "normalizer: abs_sum"
   )
   Path("abs_sum.yaml").write_text(rescored_config, encoding="utf-8")
   main([
       "reanalyze",
       "--config", "abs_sum.yaml",
       "--archive", str(run_dir),
       "--out", "rescored.json",
   ])
   rescored = json.loads(Path("rescored.json").read_text(encoding="utf-8"))
   print("backend calls:", rescored["metrics"]["executed"])
   print("abs_sum values:", [round(value, 4) for value in rescored["values"]])
   print("label:", rescored["labels"][0])

.. guide-result:: cli_reanalyze

Archives store generated text only, so embedding value functions
(``embedding_cosine``, ``embedding_euclidean``) score hash embeddings of that text
on reanalysis. The manifest's ``embedding_source`` says which embeddings the
explain run scored: ``none`` when the value function read none, ``hash`` for the
text-derived vectors and ``backend`` for model embeddings (the transformers
backend; ``embedding_model`` names the model). Reanalyzing a ``backend`` run with
an embedding value function logs a ``WARNING`` (shown with
``--log-level warning``), because the values will differ from explain; add
``--strict`` to refuse it with exit code ``2`` instead. A missing, unreadable or
older manifest or database also exits with ``2``.

Plugins
-------

``plugins`` lists every registered plugin, built-in and installed through entry
points (see :doc:`extending`):

.. code-block:: python

   main(["plugins"])

.. guide-result:: cli_plugins

Notebook
--------

``examples/cli.ipynb`` (also embedded on :doc:`../examples`) runs every
subcommand in a temporary folder and shows each exit code with its message.
