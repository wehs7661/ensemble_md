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
import os

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

def test_find_common():
    test_file1 = open(f'{input_path}/coord_swap/input_A.gro', 'r')
    test_file2 = open(f'{input_path}/coord_swap/input_B.gro', 'r')
    test_df = coordinate_swap.find_common(test_file1, test_file2, 'C2D', 'D2E')
    df = pd.read_csv(f'{input_path}/coord_swap/find_common.csv')

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
