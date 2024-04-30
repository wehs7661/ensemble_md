1. Introduction
===============
:code:`ensemble_md` is a Python package that provides methods for setting up, 
running, and analyzing GROMACS simulation ensembles. Currently, the package implements
all the necessary algorithms for running synchronous replica exchange (REX) of expanded ensembles (EE), abbreviated as
REXEE, as well as its multi-topology (MT) variation MT-REXEE. Our future work includes
implementing asynchronous REXEE and other possible variations of the REXEE method.


2. Installation
===============
2.1. Requirements
-----------------
Before installing :code:`ensemble_md`, one should have a working version of `GROMACS`_. Please refer to the GROMACS documentation for full installation instructions.
All the other pip-installable dependencies required by :code:`ensemble_md` (specified in :code:`setup.py` of the package)
will be automatically installed during the installation of the package.

.. _`GROMACS`: https://manual.gromacs.org/current/install-guide/index.html

2.2. Installation via pip
-------------------------
:code:`ensemble_md` can be installed via :code:`pip` using the following command:
::

    pip install ensemble-md 

2.3. Installation from source
-----------------------------
One can also install :code:`ensemble_md` from the source code, which is available in our
`GitHub repository`_. Specifically, one can execute the following commands:
::

    git clone https://github.com/wehs7661/ensemble_md.git
    cd ensemble_md/
    pip install .

If you would like to install the package in the editable mode, simply append the last command with the flag :code:`-e`
so that changes you make in the source code will take effect without re-installation of the package. This is particularly
useful if you would like to contribute to the development of the package. (Pull requests and issues are always welcome!)

.. _`GitHub repository`: https://github.com/wehs7661/ensemble_md.git

3. Testing
==========
3.1. Tests for the functions that do not use MPI
------------------------------------------------
Most of the tests in this package do not require MPI. To perform unit tests for these functions, execute the following command in the home directory of the project:
::

    pytest -vv --disable-pytest-warnings --cov=ensemble_md --cov-report=xml --color=yes ensemble_md/tests/

Note that the flags :code:`--cov` and :code:`--cov-report` are just for generating a coverage report and can be omitted. 
These flags require that :code:`pytest-cov` be installed. 

3.2. Tests for the functions that use MPI
-----------------------------------------
For the tests that require MPI (all implemented in :code:`tests/test_mpi_func.py`), one can use the following command:
::

    mpirun -np 4 pytest -vv --disable-pytest-warnings --cov=ensemble_md --cov-report=xml --color=yes ensemble_md/tests/test_mpi_func.py --with-mpi

Note that the flag :code:`--with-mpi` requires that :code:`pytest-mpi` be installed.
