Business and compliance applications
====================================

Shapley-style explainability connects cooperative game theory to decisions that
organizations must justify, audit, and defend. This page outlines common
deployment contexts for ``mllm-shap``-style attribution — where honest **estimand
labelling**, archived coalition records, and reproducible configs matter as much
as the numeric scores.

Theory background: :doc:`../theory/shapley_values`, :doc:`../theory/estimands`.
Hands-on usage: :doc:`getting_started`, :doc:`estimands`.

Regulated and high-stakes decisions
-----------------------------------

**Credit and lending.**
Classifiers that approve or deny loans must often explain which factors drove an
individual decision (for example, under fair-lending review). Token- or
field-level attributions on NLP features (employment text, notes, chat transcripts)
help analysts separate legitimate signals from prohibited proxies. Shapley's
**efficiency** axiom supports reconciling attributions with the total score
shift relative to a baseline.

**Insurance underwriting and claims.**
Similar needs arise when models parse free-text claims or medical narratives.
Archives that record the **estimand** (Shapley vs Banzhaf) and coalition
utilities allow actuarial and compliance teams to reproduce reviews months later.

**Anti–money laundering and fraud.**
Investigators ask *why* a transaction scored as suspicious. Attributions on
transaction narratives, merchant descriptions, or graph-derived text features
prioritize analyst time. Banzhaf-style indices can highlight **pivotal** phrases
that flip the model outcome; Shapley values provide globally consistent
decompositions for case files.

Transparency and consumer protection
------------------------------------

**Automated decision-making (GDPR Art. 22 context).**
When individuals receive solely automated decisions with legal or similar
significant effects, organizations may need *meaningful information* about the
logic involved. Coalition-based NLP explainability does not replace legal
counsel, but labelled :class:`~mllm_shap.pipeline.result.ExplainResult` objects
and :class:`~mllm_shap.pipeline.manifest.RunManifest` metadata provide a technical
audit trail: which estimand was reported, which run produced it, and which config
governed masking and scoring.

**Fairness and bias review.**
Attributions help teams test whether protected or proxy attributes dominate
predictions on representative prompts. Symmetric treatment of interchangeable
tokens (Shapley **symmetry** axiom) is a sanity check when duplicate or
templated text appears in inputs.

Enterprise NLP and LLM operations
---------------------------------

**Customer support and conversational AI.**
When a chatbot gives a wrong policy answer, attributions on the user message,
retrieved policy chunks, and system prompt identify which segments pushed the
model toward the failure. This supports prompt engineering, retrieval tuning, and
vendor escalation.

**Retrieval-augmented generation (RAG).**
Players may be retrieved passages or sentences within them. Coalition masking
simulates missing evidence; Shapley or Banzhaf scores show which sources were
load-bearing for the final answer — critical for knowledge-base maintenance and
hallucination post-mortems. The built-in partitioner makes every token a player;
to make each retrieved passage one player, register a partitioner as shown in
:doc:`extending` and set ``explanation.players`` to its name.

**Content moderation and safety.**
Safety classifiers on user-generated text benefit from explanations that isolate
triggering spans. Dummy-player behaviour (zero credit for inert tokens) filters
noise from attribution dashboards shown to human moderators.

**Internal copilots and document workflows.**
Legal, HR, and finance copilots process sensitive documents. Run archives with
typed manifests support internal audit: each archive keeps the full
:class:`~mllm_shap.ExplainConfig`, the estimand label, the base generation and every
coalition output, and :meth:`~mllm_shap.ExplainRunner.reanalyze` recomputes payoffs
and attributions from them without calling the model again.

Model risk management
---------------------

Banking supervisors (for example OCC SR 11-7 and related guidance) expect
institutions to validate models, document assumptions, and monitor drift.
Explainability is part of the model risk toolkit — not a substitute for
validation, but a required artifact for many NLP deployments.

Practices that align with ``mllm-shap`` design:

- **Versioned configs** via :class:`~mllm_shap.ExplainConfig` and YAML archives
- **Explicit estimands** so validation reports state whether Shapley or Banzhaf
  was used
- **Separation of concerns** — estimators sample coalitions; estimand aggregators
  apply weights; value functions score outputs — mirroring independent validation
  of each pipeline stage

Choosing estimands for stakeholders
-----------------------------------

+------------------+-----------------------------+-----------------------------+
| Audience         | Often prefers               | Why                         |
+==================+=============================+=============================+
| Risk & compliance| Shapley                     | Efficiency: scores sum to   |
| committees       |                             | total utility change        |
+------------------+-----------------------------+-----------------------------+
| Fraud / security | Banzhaf or Shapley          | Banzhaf highlights pivotal  |
| analysts         |                             | triggers; Shapley for cases |
+------------------+-----------------------------+-----------------------------+
| Product / UX     | Shapley                     | Easier global narrative:    |
| teams            |                             | "how much each part moved   |
|                  |                             | the answer"                 |
+------------------+-----------------------------+-----------------------------+
| Research         | Both, clearly labelled      | Non-additive games show     |
| publications     |                             | materially different values |
+------------------+-----------------------------+-----------------------------+

Report the estimand in every deliverable. Mixing labels invalidates cross-run
comparisons and regulatory narratives.

Operational checklist
---------------------

Before production use:

1. Define players (tokens, spans, chunks) and absence policy for your use case.
2. Choose **estimand** (Shapley vs Banzhaf) to match stakeholder questions.
3. Keep run archives (manifest, base generation and coalition outputs) for
   reproducibility; attributions are recomputed from them with ``reanalyze``.
4. Document value functions and baselines (:math:`v(\emptyset)`) in run configs.
5. Validate on known toy games (majority rule, additive cases) before scaling.

Further reading
---------------

- Theory: :doc:`../theory/shapley_values`
- Estimand choice: :doc:`../theory/estimands`
- Configuration: :doc:`config`
- API: :doc:`../api`

References
----------

- Board of Governors of the Federal Reserve System. (2011). SR 11-7: Guidance on
  model risk management.
  `Federal Reserve SR 11-7 <https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.htm>`__
- Regulation (EU) 2016/679 (GDPR), Article 22 — automated individual decision-making.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *Advances in Neural Information Processing Systems*, 30.
  `arXiv:1705.07874 <https://arxiv.org/abs/1705.07874>`__
- Fryer, D. V., Strümke, I., & Nguyen, H. D. (2021). Shapley values for feature
  selection: The good, the bad, and the axioms. *IEEE Access*, 9, 144352–144360.
  `arXiv:2102.10936 <https://arxiv.org/abs/2102.10936>`__
