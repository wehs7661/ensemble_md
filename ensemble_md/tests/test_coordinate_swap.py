####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
"""
Unit tests for the module coordinate_swap.py.
"""
from ensemble_md.utils import coordinate_swap

def test_get_dimenstion():
    test_file1 = open('ensemble_md/tests/data/coord_swap/input_A.gro', 'r')
    test_file2 = open('ensemble_md/tests/data/coord_swap/input_B.gro', 'r')
    assert coordinate_swap.get_dimensions(test_file1) == []
    assert coordinate_swap.get_dimensions(test_file2) == []

