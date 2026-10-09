Shapley values and axioms
=========================

The **Shapley value** is the classical fair-division solution for cooperative
games. In explainability it underpins SHAP-style feature attributions when
coalition payoffs are aggregated with Shapley weights. This page states the
definition, the four defining axioms, and the uniqueness result from Shapley
(1953).

For Banzhaf indices and estimand comparison in ML pipelines, see
:doc:`estimands`. For business deployment contexts, see
:doc:`../guides/applications`.

Definition
----------

For a cooperative game with player set :math:`N`, :math:`|N| = n`, and
characteristic function :math:`v`, the Shapley value :math:`\phi_i(v)` for
player :math:`i \in N` is:

.. math::

   \phi_i(v) = \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!\,(n-|S|-1)!}{n!}
   \bigl(v(S \cup \{i\}) - v(S)\bigr)

Interpretation: imagine players join a coalition in random order, uniformly over
all :math:`n!` permutations. The Shapley value is player :math:`i`'s expected
marginal contribution when joining.

For coalition size :math:`k = |S|` (excluding :math:`i`), the weight is:

.. math::

   w_{\mathrm{Shapley}}(k, n) = \frac{k!\,(n-k-1)!}{n!}

Implemented in :class:`~nlp_shap.estimation.estimands.shapley.ShapleyAggregator`.

The four axioms
---------------

Shapley (1953) characterized :math:`\phi` as the **unique** allocation rule
satisfying four axioms on the space of cooperative games. Let :math:`\phi(v) =
(\phi_1(v), \dots, \phi_n(v))` denote the allocation vector for game :math:`v`.

Efficiency
~~~~~~~~~~

The entire worth of the grand coalition is distributed. For games normalized
so that :math:`v(\emptyset) = 0`:

.. math::

   \sum_{i \in N} \phi_i(v) = v(N)

Payoffs in ``nlp-shap`` are similarities to the base generation, so the empty
prompt usually scores above zero. The library therefore uses the convention

.. math::

   \sum_{i \in N} \phi_i(v) = v(N) - v(\emptyset)

which is the axiom applied to the normalized game
:math:`v'(S) = v(S) - v(\emptyset)`: subtracting a constant from every payoff
leaves every marginal contribution, and hence every Shapley value, unchanged.
Each result reports both terms as
:attr:`~nlp_shap.ExplainResult.grand_value` (:math:`v(N)`) and
:attr:`~nlp_shap.ExplainResult.base_value` (:math:`v(\emptyset)`).

**Explainability reading:** attributions sum to the change in model utility
between the full prompt and the fully masked prompt. On the full coalition table
(``exact``, or any estimator at ``budget.fraction: 1.0``) the identity holds to
floating-point precision and is the first consistency check in an audit; sampled
estimates approach it as the budget grows.

Symmetry
~~~~~~~~

If players :math:`i` and :math:`j` make identical marginal contributions in
every coalition — that is,
:math:`v(S \cup \{i\}) = v(S \cup \{j\})` for all
:math:`S \subseteq N \setminus \{i, j\}` — then:

.. math::

   \phi_i(v) = \phi_j(v)

**Explainability reading:** interchangeable tokens (e.g. duplicate padding) receive
equal credit.

Dummy player (null player)
~~~~~~~~~~~~~~~~~~~~~~~~~~

If player :math:`i` never changes the payoff — formally
:math:`v(S \cup \{i\}) = v(S)` for all :math:`S \subseteq N \setminus \{i\}` —
then:

.. math::

   \phi_i(v) = 0

**Explainability reading:** tokens that do not affect the scored output receive
zero attribution.

Additivity
~~~~~~~~~~

For two games :math:`v` and :math:`w` on the same player set, define
:math:`(v + w)(S) = v(S) + w(S)`. Then:

.. math::

   \phi_i(v + w) = \phi_i(v) + \phi_i(w)

**Explainability reading:** attributions decompose additively when utilities are
sums of independent scoring components (for example, multi-objective value
functions).

Uniqueness theorem
------------------

**Theorem (Shapley, 1953).** There exists exactly one allocation rule
:math:`\phi` that satisfies efficiency, symmetry, the dummy-player axiom, and
additivity. It is given by the formula above.

This uniqueness result is why SHAP (Lundberg & Lee, 2017) argues for Shapley-based
additive feature attributions under local accuracy and missingness constraints:
within the class of additive explanation models, the Shapley kernel is the unique
choice satisfying a parallel set of desirable properties.

Connection to SHAP in machine learning
--------------------------------------

Lundberg & Lee (2017) cast feature attribution as a cooperative game where
features are players and :math:`v(S)` is the model output (or a transform) when
only feature subset :math:`S` is observed. Under **local accuracy** (attributions
sum to the prediction difference from a baseline) and **missingness** (absent
features do not contribute), the Shapley value yields the SHAP explanation.

Important distinction for ``nlp-shap``:

- **SHAP** names a family of explanation methods tied to Shapley allocations.
- An **estimand** is the quantity your pipeline actually estimates. Averaging
  the marginal contributions of sampled coalitions uniformly targets the
  **Banzhaf index**; averaging them within each coalition size first and then
  across sizes (a size-stratified mean) targets the **Shapley value**. The
  ``nlp-shap`` aggregators do the latter for Shapley and the former for Banzhaf
  — see :doc:`estimands` and :doc:`approximation`.

When Shapley values are appropriate
-----------------------------------

Shapley allocations are a strong default when you need:

- **Efficiency** for reconciliation with total model utility
- **Fair treatment** of symmetric players
- **Additive decomposition** across independent utility components
- **Regulatory narratives** that reference the standard game-theoretic fairness
  axioms

They are not always the right estimand when you want **pivotal-player**
(voting-power) interpretations; Banzhaf indices emphasize how often a player
*swings* a coalition from losing to winning. See :doc:`estimands` for when the
two diverge.

References
----------

- Shapley, L. S. (1953). A value for *n*-person games. In H. W. Kuhn & A. W.
  Tucker (Eds.), *Contributions to the Theory of Games II* (pp. 307–317).
  Princeton University Press.
  `DOI:10.1515/9781400881970-018 <https://doi.org/10.1515/9781400881970-018>`__
- Shapley, L. S. (1952). A value for *n*-person games (RAND Paper P-295). RAND
  Corporation.
  `RAND P-295 <https://www.rand.org/pubs/papers/P295.html>`__
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *Advances in Neural Information Processing Systems*, 30.
  `ML Anthology <https://mlanthology.org/neurips/2017/lundberg2017neurips-unified/>`__;
  `arXiv:1705.07874 <https://arxiv.org/abs/1705.07874>`__
- Roth, A. E. (Ed.). (1988). *The Shapley Value: Essays in Honor of Lloyd S.
  Shapley*. Cambridge University Press.
- Young, H. P. (1985). Monotonic solutions of cooperative games.
  *International Journal of Game Theory*, 14(2), 65–72.
  `DOI:10.1007/BF01769885 <https://doi.org/10.1007/BF01769885>`__
