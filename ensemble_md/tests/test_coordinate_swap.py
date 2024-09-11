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
Unit tests for the module test_coordainte_swap.py.
"""
import os
import numpy as np
from ensemble_md.utils import coordinate_swap

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def test_get_dimensions():
    gro = os.path.join(input_path, 'sys.gro')
    f = open(gro, 'r')
    lines = f.readlines()
    f.close()
    vec = coordinate_swap.get_dimensions(lines)
    assert vec == [3.32017, 3.32017, 2.34772, 0.00000, 0.00000, 0.00000, 0.00000, 1.66009, 1.66009]

    # Write a flat file with cubic box dimensions
    f = open('test.gro', 'w')
    f.write('test\n')
    f.write('    1.00000    2.00000    3.00000\n')
    f.close()

    f = open('test.gro', 'r')
    lines = f.readlines()
    f.close()
    vec = coordinate_swap.get_dimensions(lines)
    assert vec == [1.0, 2.0, 3.0]

    os.remove('test.gro')


def test_compute_angle():
    coords_1 = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0])
    ]
    coords_2 = coords_1[-1::-1]
    coords_3 = [coords_1[1], coords_1[0], coords_1[2]]

    assert np.isclose(coordinate_swap.compute_angle(coords_1), np.pi / 4)
    assert np.isclose(coordinate_swap.compute_angle(coords_2), np.pi / 4)
    assert np.isclose(coordinate_swap.compute_angle(coords_3), np.pi / 2)
