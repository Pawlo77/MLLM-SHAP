API reference
=============

``mllm-shap`` exposes a small, typed public surface. Import the symbols below
from :mod:`mllm_shap`; the module pages document every class and function where it
is defined.

Public exports
--------------

The table is generated from ``mllm_shap.__all__`` at build time, so it always lists
what ``from mllm_shap import ...`` accepts.

.. public-exports::

Modules
-------

.. toctree::
   :maxdepth: 1

   api/domain
   api/protocols
   api/plugins
   api/pipeline
   api/cli
   api/errors
   api/masking
   api/runtime
   api/estimation
   api/value
   api/backends
   api/viz
   api/alignment

Package-level objects
---------------------

.. automodule:: mllm_shap
   :no-members:

.. autofunction:: mllm_shap.configure_logging

.. py:data:: mllm_shap.__version__
   :type: str

   Installed package version, read from the distribution metadata
   (``0.0.0+unknown`` when the package is not installed).
