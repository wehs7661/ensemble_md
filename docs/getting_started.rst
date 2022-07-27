Introduction
============
:code:`ensemble_md` is a Python package providing methods for running 
GROMACS simulation ensembles. Currently, we have implemented the method 
of ensemble of expanded ensemble. Other methods such as ensemble of alchemical
metadynamics will be under development in the future. In our present implementation, 
`gmxapi`_, which is a higher level Python API of GROMACS, is used to launch GROMACS 
commands, but we will switch to `SCALE-MS`_ for this purpose in the future. 


.. _`gmxapi`: https://manual.gromacs.org/current/gmxapi/
.. _`SCALE-MS`: https://scale-ms.readthedocs.io/en/latest/


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


