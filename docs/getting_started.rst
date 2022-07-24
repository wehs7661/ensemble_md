Introduction
============
:code:`ensemble_md` is a Python package providing methods for running 
GROMACS simulation ensembles, including ensemble of expanded ensemble 
and ensemble of alchemical metadynamics. The former is our main focus 
in our current phase, while the latter will be under development in 
the future. Currently, :code:`ensemble_md` uses a higher level Python API 
of GROMACS, :code:`gmxapi`, to launch GROMACS simulations and access relevant 
files programmatically as needed. We will switch to :code:`scale-ms` for 
this purpose in the future. 


Installation
============
The package has not been published to PyPI, but can be installed from our
`github repository`_ using the following commands:
::

    git clone https://github.com/wehs7661/ensemble_md.git
    cd ensemble_md/
    pip install -e .

Note that this package requries a bunch of other Python packages to be installed,
including NumPy, pymbar, natsort, and argparse. For the full list please
check :code:`requirements.txt`.

.. _`github repository`: https://github.com/wehs7661/ensemble_md.git


