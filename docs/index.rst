.. ensemble_md documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ensemble_md's documentation!
=========================================================
``ensemble_md`` is a Python package providing methods for running 
GROMACS simulation ensembles, including ensemble of expanded ensemble 
and ensemble of alchemical metadynamics. The former is our main focus 
in our current phase, while the latter will be under development in 
the future.

.. toctree::
   getting_started
   :maxdepth: 2
   :caption: Getting started:

.. toctree::
   theory
   :maxdepth: 2
   :caption: Theory:

.. toctree::
   simulations
   :maxdepth: 2
   :caption: Launching REXEE simulations:
   
.. toctree::
   examples/run_REXEE
   examples/analyze_REXEE
   examples/run_REXEE_modify_inputs
   :maxdepth: 2
   :caption: Tutorials:

.. toctree::
   api/api_replica_exchange_EE.rst
   api/api_analysis
   api/api_utils
   :maxdepth: 2
   :caption: API documentation:

.. toctree::
   references
   :maxdepth: 1
   :caption: References:

Others
======

* :ref:`genindex`
