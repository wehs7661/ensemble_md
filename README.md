Ensemble Molecular Dynamics
==============================
[//]: # (Badges)
[![wehs7661](https://circleci.com/gh/wehs7661/ensemble_md.svg?style=shield)](https://app.circleci.com/pipelines/github/wehs7661/ensemble_md?branch=master)
[![codecov](https://codecov.io/gh/wehs7661/ensemble_md/branch/master/graph/badge.svg)](https://app.codecov.io/gh/wehs7661/ensemble_md/tree/master)
[![Documentation Status](https://readthedocs.org/projects/ensemble-md/badge/?version=latest)](https://ensemble-md.readthedocs.io/en/latest/?badge=latest)
[![GitHub Actions Lint Status](https://github.com/wehs7661/ensemble_md/actions/workflows/lint.yaml/badge.svg)](https://github.com/wehs7661/ensemble_md/actions/workflows/lint.yaml)
[![PyPI version](https://badge.fury.io/py/ensemble-md.svg)](https://badge.fury.io/py/ensemble-md)
[![python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11-4BC51D.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![DOI](https://img.shields.io/badge/DOI-10.1021/acs.jctc.4c00484-4BC51D)](https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.4c00484)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Downloads](https://static.pepy.tech/badge/ensemble-md)](https://pepy.tech/project/ensemble-md)


`ensemble_md` is a Python package that provides methods for setting up, running, and analyzing GROMACS simulation ensembles. Currently, the package implements all the necessary algorithms for running synchronous replica exchange (REX) of expanded ensembles (EE), abbreviated as REXEE, as well as its multi-topology (MT) variation, MT-REXEE. Our future work includes implementing asynchronous REXEE and other possible variations of the REXEE method. For installation instructions, theory overview, tutorials, and API references, please visit the [documentation](https://ensemble-md.readthedocs.io/en/latest/?badge=latest) and our [JCTC paper](https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.4c00484).

### Reference
If you use any components of the Python package `ensemble_md` or the REXEE method in your research, please cite the following paper:

Hsu, W. T., & Shirts, M. R. (2024). Replica Exchange of Expanded Ensembles: A Generalized Ensemble Approach with Enhanced Flexibility and Parallelizability. *Journal of Chemical Theory and Computation*. doi: [10.1021/acs.jctc.4c00484](https://doi.org/10.1021/acs.jctc.4c00484)

### Copyright

Copyright (c) 2022, Wei-Tse Hsu


### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
