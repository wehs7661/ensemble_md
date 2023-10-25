1. Introduction
===============
:code:`ensemble_md` is a Python package that provides methods for setting up, 
running, and analyzing GROMACS simulation ensembles. The current implementation is
mainly for synchronous replica exchange (REX) of expanded ensemble (EE), abbreviated as
REXEE. In the future, we will develop methods like asynchronous REXEE, or multi-topology REXEE.
In the current implementation, the module :code:`subprocess`
is used to launch GROMACS commands, but we will switch to `SCALE-MS`_ for this purpose
in the future when possible.


.. _`SCALE-MS`: https://scale-ms.readthedocs.io/en/latest/


2. Installation
===============
2.1. Requirements
-----------------
Before installing :code:`ensemble_md`, one should have working versions of `GROMACS`_. Please refer to the linked documentation for full installation instructions.
All the other pip-installable dependencies of :code:`ensemble_md` (specified in :code:`setup.py` of the package)
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
`github repository`_. Specifically, execute the following commands:
::

    git clone https://github.com/wehs7661/ensemble_md.git
    cd ensemble_md/
    pip install .

If you are interested in contributing to the project, append the 
last command with the flag :code:`-e` to install the project in the editable mode 
so that changes you make in the source code will take effects without re-installation of the package. 
(Pull requests to the project repository are welcome!)

.. _`github repository`: https://github.com/wehs7661/ensemble_md.git

3. Testing
==========
To perform unit tests for this package, execute the following command in the home directory of the project:
::

    pytest -vv --disable-pytest-warnings --cov=ensemble_md --cov-report=xml --color=yes ensemble_md/tests/

or 

::

    python -m pytest -vv --disable-pytest-warnings --cov=ensemble_md --cov-report=xml --color=yes ensemble_md/tests/

Note that the flags :code:`--cov` and :code:`--cov-report` require that :code:`pytest-cov` be installed. 