Citing mllm-shap
================

If you use ``mllm-shap`` in research, please cite the software. The repository's
`CITATION.cff <https://github.com/Pawlo77/MLLM-SHAP/blob/main/CITATION.cff>`_ is
the source of truth; GitHub's "Cite this repository" button renders it as APA or
BibTeX.

BibTeX
------

.. code-block:: bibtex

   @software{pozorski_mllm_shap,
     author  = {Pozorski, Paweł},
     title   = {mllm-shap: Shapley-value explainability for conversational
                language models},
     url     = {https://github.com/Pawlo77/MLLM-SHAP},
     license = {Apache-2.0},
     note    = {Python package, version as used in your work}
   }

Add the ``version`` field with the release you used
(``python -c "import mllm_shap; print(mllm_shap.__version__)"``) and record the run
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
