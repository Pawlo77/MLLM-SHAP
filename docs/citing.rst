Citing nlp-shap
===============

If you use ``nlp-shap`` in research, please cite the software. The repository's
`CITATION.cff <https://github.com/Pawlo77/nlp-shap/blob/main/CITATION.cff>`_ is
the source of truth; GitHub's "Cite this repository" button renders it as APA or
BibTeX.

BibTeX
------

.. code-block:: bibtex

   @software{pozorski_nlp_shap,
     author  = {Pozorski, Paweł},
     title   = {nlp-shap: Shapley-value explainability for conversational
                language models},
     url     = {https://github.com/Pawlo77/nlp-shap},
     license = {Apache-2.0},
     note    = {Python package, version as used in your work}
   }

Add the ``version`` field with the release you used
(``python -c "import nlp_shap; print(nlp_shap.__version__)"``) and record the run
manifest of each reported result, which pins the estimand, estimator, value
function and model.

Citation File Format
--------------------

.. literalinclude:: ../CITATION.cff
   :language: yaml

Methods
-------

Please also cite the methods your results depend on. The theory pages list the
original references:

- Shapley value and axioms: :doc:`theory/shapley_values`
- Banzhaf index and estimand choice: :doc:`theory/estimands`
- Sampling estimators (Monte Carlo, complementary contributions, Neyman
  allocation): :doc:`theory/approximation`
