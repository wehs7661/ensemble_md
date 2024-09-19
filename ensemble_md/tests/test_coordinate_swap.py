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
import numpy as np
import pandas as pd

def test_get_dimenstion():
    test_file1 = open('ensemble_md/tests/data/coord_swap/input_A.gro', 'r')
    test_file2 = open('ensemble_md/tests/data/coord_swap/input_B.gro', 'r')
    assert coordinate_swap.get_dimensions(test_file1) == [2.74964, 2.74964, 2.74964]
    assert coordinate_swap.get_dimensions(test_file2) == [2.74243, 2.74243, 2.74243]

def test_find_common():
    test_file1 = open('ensemble_md/tests/data/coord_swap/input_A.gro', 'r')
    test_file2 = open('ensemble_md/tests/data/coord_swap/input_B.gro', 'r')
    test_df = coordinate_swap.find_common(test_file1, test_file2, 'C2D', 'D2E')
    df = pd.read_csv('ensemble_md/tests/data/coord_swap/find_common.csv')

    for index, row in df.iterrows():
        test_row = test_df[test_df['Name'] == row['Name']]
        assert row['Atom Name Number'] == int(test_row['Atom Name Number'].to_list()[0])
        assert row['Element'] == test_row['Element'].to_list()[0]
        assert row['Direction'] == test_row['Direction'].to_list()[0]
        assert row['Swap'] == test_row['Swap'].to_list()[0]
        assert row['File line'] == int(test_row['File line'].to_list()[0])
        assert row['Final Type'] == test_row['Final Type'].to_list()[0]

def test_rotate_point_around_axis():
    initial_point = np.array([0.16, 0.19, -0.05])
    vertex = np.array([0, 0, 0])
    axis = np.array([0.15, 0.82, 0.14])
    angle = 0.13
    rotated_point = [0.1693233, 0.18548463, -0.0335421]
    assert coordinate_swap.rotate_point_around_axis(initial_point, vertex, axis, angle) == rotated_point

def test_find_rotation_angle():
    initial_point = np.array([0.16, 0.19, -0.05])
    vertex = np.array([0, 0, 0])
    axis = np.array([0.15, 0.82, 0.14])
    rotated_point = [0.1693233, 0.18548463, -0.0335421]
    angle = 0.13
    test_angle = coordinate_swap.find_rotation_angle(initial_point, vertex, rotated_point, axis)
    assert np.isclose(angle, test_angle, 10**(-5))

def test_compute_angle():
    vec1_start = [0.13, 0.15, 0.16]
    vec1_end = [0.16, 0.18, 0.17]
    vec2_start = [0.13, 0.15, 0.16]
    vec2_end = [0.11, 0.23, 0.05]
    assert np.isclose(coordinate_swap.compute_angle([vec1_start, vec1_end, vec2_start, vec2_end]), 0.991836017536949, atol=10**(-4))