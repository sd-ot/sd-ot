# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# python -m pip install --upgrade --no-cache-dir pip setuptools
# python -m pip install --upgrade --no-cache-dir pillow mock==1.0.1 alabaster>=0.7,<0.8,!=0.7.5 commonmark==0.9.1 recommonmark==0.5.0 sphinx sphinx-rtd-theme readthedocs-sphinx-ext<2.3
# cat docs/source/conf.py
# python -m sphinx -T -E -b html -d _build/doctrees -D language=en . $READTHEDOCS_OUTPUT/html
# python -m sphinx -T -E -b readthedocssinglehtmllocalmedia -d _build/doctrees -D language=en . 