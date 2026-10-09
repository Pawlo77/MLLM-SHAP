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

   from nlp_shap import (
       ConversationSnapshot,
       ExplainConfig,
       ExplainRunner,
       Message,
       Role,
       Turn,
       render_attribution,
   )
   from nlp_shap.masking import TokenPartitioner

   snapshot = ConversationSnapshot.from_turns((
       Turn(messages=(Message(role=Role.USER, text="refund within thirty days"),)),
   ))
   config = ExplainConfig.model_validate({
       "backend": {"kind": "mock", "model_id": "stub"},
       "explanation": {
           "estimator": "exact",
           "estimand": "shapley",
           "value_fn": "tfidf_cosine",
           "absence_policy": "pad",
           "archive": {"path": ""},
       },
   })
   output = ExplainRunner(config).explain_sync(snapshot)
   player_set = TokenPartitioner().partition(snapshot)

   bar = render_attribution(output, snapshot, player_set, renderer="token_bar")
   text = render_attribution(output, snapshot, player_set, renderer="token_text")
   bar.savefig("attribution.png", dpi=150, bbox_inches="tight")

.. guide-result:: visualization_bar
   :caption: Bar chart (``token_bar``)
   :figure:
   :alt: Horizontal token attribution bar chart

.. guide-result:: visualization_text
   :caption: Inline tokens (``token_text``)
   :figure:
   :alt: Inline colored token attribution

``player_set`` must be the partition the run used; with ``players: tokens`` that
is :class:`~nlp_shap.masking.TokenPartitioner`. Pass ``title=`` to label a figure
and ``registry=`` to use a renderer registered in your own
:class:`~nlp_shap.PluginRegistry`.

Jupyter HTML
------------

In a notebook, :func:`~nlp_shap.viz.display_attribution_html` shows the colored
tokens inline without matplotlib, and :func:`~nlp_shap.render_attribution_html`
returns the same fragment as a string for reports:

.. code-block:: python

   from nlp_shap import render_attribution_html

   html = render_attribution_html(output, snapshot, player_set, title="Refund")
   print(type(html).__name__, all(word in html for word in ("refund", "days")))

.. guide-result:: visualization_html

In Jupyter, ``display_attribution_html(output, snapshot, player_set)`` renders it
directly.

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
