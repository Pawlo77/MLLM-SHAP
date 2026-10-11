# Minimal makefile for Sphinx documentation
#
# Strict by default: warnings are errors (-W), every warning is reported
# (--keep-going) and every cross-reference must resolve (-n, see conf.py).

SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

NOTEBOOK_SOURCES = explain_pipeline.ipynb archive_reanalyze.ipynb cli.ipynb plugins_extending.ipynb estimands_toy_game.ipynb masking_views.ipynb runtime_core.ipynb exact_estimation.ipynb estimator_comparison.ipynb attribution_viz.ipynb text_explain_e2e.ipynb audio_alignment.ipynb mock_audio_backend.ipynb audio_explain_transformers.ipynb

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean copy-notebooks notebook-figures render-guide-outputs export-knowledge linkcheck

clean:
	rm -rf "$(BUILDDIR)" notebooks _static/notebook_figures

copy-notebooks:
	@mkdir -p notebooks
	@for nb in $(NOTEBOOK_SOURCES); do \
		cp ../examples/$$nb notebooks/$$nb; \
	done

notebook-figures:
	@cd .. && uv run python docs/scripts/notebook_figures.py

render-guide-outputs:
	@cd .. && uv run python docs/scripts/render_guide_outputs.py

export-knowledge:
	@cd .. && uv run python scripts/export_knowledge.py

# External links only; doi.org is skipped (rate limits), see linkcheck_ignore.
linkcheck: copy-notebooks notebook-figures export-knowledge
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile copy-notebooks notebook-figures render-guide-outputs export-knowledge
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
