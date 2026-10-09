Cooperative games
=================

``nlp-shap`` explains model behaviour by treating input units (tokens, segments,
or other **players**) as participants in a **cooperative game**. Cooperative game
theory studies how a total payoff :math:`v(N)` should be distributed among
players when they can form coalitions. In explainability, the payoff comes from
scoring model outputs under different input subsets.

This page establishes the game-theoretic objects. For Shapley axioms and
uniqueness, see :doc:`shapley_values`. For Shapley vs Banzhaf in ML pipelines,
see :doc:`estimands`. For deployment contexts, see :doc:`../guides/applications`.

Players, coalitions, and payoffs
--------------------------------

A cooperative game is defined by:

- a finite player set :math:`N = \{1, \dots, n\}`
- a **characteristic function** :math:`v : 2^N \to \mathbb{R}`

The value :math:`v(S)` is the worth of coalition :math:`S \subseteq N`. By
convention :math:`v(\emptyset)` is the **empty-coalition reference** (often
:math:`0` or a baseline utility). The **grand coalition** :math:`N` represents
the fully observed input.

In NLP explainability:

- **Players** may be tokens, spans, retrieved chunks, or messages.
- **Coalitions** specify which players are *present*; absent players are removed
  or replaced according to an absence policy (implemented in later pipeline
  stages).
- **Payoffs** :math:`v(S)` come from a **value function** applied to model
  outputs when only coalition :math:`S` is visible to the model.

Marginal contributions
----------------------

For player :math:`i` and coalition :math:`S \subseteq N \setminus \{i\}` the
**marginal contribution** is:

.. math::

   \Delta_i(S) = v(S \cup \{i\}) - v(S)

Attribution methods differ in how they **weight** marginal contributions across
coalitions. The Shapley value uses size-dependent factorial weights; the Banzhaf
index uses uniform weights over :math:`S` (see :doc:`estimands`).

Game classes relevant to explainability
---------------------------------------

**Additive games.**
If :math:`v(S) = \sum_{i \in S} a_i` for fixed coefficients :math:`a_i`, then
every reasonable linear index recovers :math:`a_i`. Many tabular explainability
examples are additive; language games are usually **not**, because token
interactions are context-dependent.

**Threshold / voting games.**
Payoff is :math:`1` when at least :math:`t` players are present and :math:`0`
otherwise. These games illustrate why **estimand choice matters**: Shapley and
Banzhaf assign different credit on identical coalition samples.

**Non-additive LLM games.**
When :math:`v(S)` is computed by running a language model on a masked prompt,
payoffs depend on syntax, coreference, and retrieval context. The characteristic
function is typically neither additive nor symmetric across tokens. Honest
**estimand labelling** and archived coalition records become essential for audit
and reproduction.

Mapping to ``nlp-shap`` types
-----------------------------

+---------------------------+--------------------------------------------------+
| Game-theoretic object     | ``nlp-shap`` type                                |
+===========================+==================================================+
| Player set :math:`N`      | :class:`~nlp_shap.PlayerSet`                     |
+---------------------------+--------------------------------------------------+
| Coalition :math:`S`       | :class:`~nlp_shap.CoalitionMask`                 |
+---------------------------+--------------------------------------------------+
| Conversation under study  | :class:`~nlp_shap.ConversationSnapshot`          |
+---------------------------+--------------------------------------------------+
| Game :math:`(N, v)`       | :class:`~nlp_shap.CooperativeGame` (+ payoffs)   |
+---------------------------+--------------------------------------------------+
| Aggregated attributions   | :class:`~nlp_shap.pipeline.result.ExplainResult` |
+---------------------------+--------------------------------------------------+

The :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator` and
:class:`~nlp_shap.estimation.estimands.banzhaf.BanzhafAggregator` classes
aggregate precomputed coalition masks and payoffs :math:`v(S)`; they do not run
model inference themselves.

Multilinear extensions (optional)
---------------------------------

Owen (1972) introduced **multilinear extensions**, connecting cooperative games
to expectations over random coalitions. Monte Carlo explainability estimators
sample coalitions and average marginal contributions; the **estimand** determines
whether Shapley or Banzhaf weights are applied to those samples. See
:doc:`shapley_values` for the connection to the Shapley formula.

References
----------

- von Neumann, J., & Morgenstern, O. (1944). *Theory of Games and Economic
  Behavior*. Princeton University Press.
- Shapley, L. S. (1953). A value for *n*-person games. In H. W. Kuhn & A. W.
  Tucker (Eds.), *Contributions to the Theory of Games II* (Vol. 28, pp.
  307–317). Princeton University Press.
  `DOI:10.1515/9781400881970-018 <https://doi.org/10.1515/9781400881970-018>`__
- Owen, G. (1972). Multilinear extensions of games. *Management Science*, 18(5
  Part 2), 64–79.
  `DOI:10.1287/mnsc.18.5.64 <https://doi.org/10.1287/mnsc.18.5.64>`__
- Roth, A. E. (Ed.). (1988). *The Shapley Value: Essays in Honor of Lloyd S.
  Shapley*. Cambridge University Press.
