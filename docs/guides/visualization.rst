Visualization
=============

Per-token Shapley and Banzhaf attributions are numeric tuples in
:class:`~nlp_shap.ExplainRunOutput`. Renderer plugins turn those
values into review-ready figures for notebooks, reports, and stakeholder decks.

Renderers
---------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Plugin key
     - Output
   * - ``token_text``
     - SHAP-style inline colored tokens (red / blue diverging highlights)
   * - ``token_bar``
     - Horizontal bar chart with SHAP palette, value labels, and sign legend

Install the optional visualization extra:

.. code-block:: bash

   pip install "nlp-shap[viz]"

The renderers follow the `SHAP <https://github.com/shap/shap>`_ palette: warm red
for positive attributions and cool blue for negative attributions, with soft
white-centered highlights for inline tokens. Figures use a Seaborn-backed clean
theme (muted grid, no chart junk).

Quick start
-----------

.. code-block:: python

   from nlp_shap import render_attribution

   output = ExplainRunner(config).explain_sync(snapshot)
   bar = render_attribution(output, snapshot, output.player_set, renderer="token_bar")
   text = render_attribution(output, snapshot, output.player_set, renderer="token_text")
   bar.savefig("attribution.png", dpi=150, bbox_inches="tight")

Pass the run's ``output.player_set`` so labels match the players, ``title=`` to
label a figure and ``registry=`` to use a renderer registered in your own
:class:`~nlp_shap.PluginRegistry`. The figures below come from
:doc:`../notebooks/attribution_viz`, which explains three inputs with
``Qwen/Qwen2.5-1.5B-Instruct`` and ``tfidf_cosine``.

Words of a prompt
-----------------

.. figure:: /_static/notebook_figures/viz_tokens_text.png
   :alt: The refund prompt with each word shaded red or blue by its value.

   ``token_text``: the words in reading order; "yes", "no.", "Refund?" and
   "laptop" push the answer toward the original one, "opened it." pulls against it.

.. figure:: /_static/notebook_figures/viz_tokens_bar.png
   :alt: Bar chart of the 13 word attributions, sorted by size.

   ``token_bar``: the same values sorted by absolute size, each bar labelled
   with its signed value.

Messages of a conversation
--------------------------

With ``players: messages`` each message is a player labelled with its role,
speaker and position; ``title=`` names the figure.

.. figure:: /_static/notebook_figures/viz_messages_bar.png
   :alt: Bar chart of five conversation messages labelled with role and speaker.

   A five-message support thread: Alice's order and Bob's two messages carry the
   answer, the assistant's greeting and Alice's "used it once" matter less.

:func:`~nlp_shap.viz.plot_conversation` keeps the thread's shape: each message
under its role, speaker and position, with every player shaded where it stands
and text that is no player in grey. :func:`~nlp_shap.viz.render_conversation_html`
returns the same view as HTML. It works for message and token players alike.

.. figure:: /_static/notebook_figures/viz_thread_messages.png
   :alt: The thread drawn message by message, each message shaded by its value.

   ``plot_conversation`` with ``players: messages``.

With ``players: tokens`` and ``plugin_options: {tokens: {roles: [user]}}`` only the
users' words are players; the other messages stay present and appear grey.

.. figure:: /_static/notebook_figures/viz_thread_tokens.png
   :alt: The thread with the users' words shaded and the assistant message grey.

   Per-word attributions of the users' messages: "Can", "Answer" and "yes" in
   Bob's question decide; the assistant's greeting is context, not a player.

Spoken words
------------

For audio players (``players: sgpa_segments``, :doc:`alignment`),
:func:`~nlp_shap.viz.plot_audio_segments` colours each word's stretch of audio by
its value over the waveform and its energy.

.. figure:: /_static/notebook_figures/viz_audio.png
   :alt: Waveform of a spoken sentence with word segments coloured by value.

   "curiosity" and "I" shape the model's reading of the speaker's mood most;
   "at" counts slightly against it.

Jupyter HTML
------------

In a notebook, :func:`~nlp_shap.viz.display_attribution_html` shows the colored
tokens inline without matplotlib, and :func:`~nlp_shap.render_attribution_html`
returns the same fragment as a string for reports:

.. code-block:: python

   from nlp_shap import display_attribution_html, render_attribution_html

   display_attribution_html(output, snapshot, output.player_set)
   html = render_attribution_html(output, snapshot, output.player_set, title="Refund")

Color scale
-----------

Positive attributions use SHAP red (``#FF0D57``); negative attributions use SHAP
blue (``#1E88E5``). Inline token backgrounds fade from white toward the signed
color, matching the ``shap.plots.text`` emphasis pattern. Bar charts annotate
each token with its signed value.

Further reading
---------------

- ``examples/attribution_viz.ipynb`` and ``examples/text_explain_e2e.ipynb``
  (embedded on :doc:`../examples`)
- Generative backends: :doc:`backends`
- API reference: :doc:`../api/viz`
