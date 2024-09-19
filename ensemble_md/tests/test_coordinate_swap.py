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
import mdtraj as md

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


def test_R2D_D2R_miss():
    nameA_list = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8', 'H9', 'H10', 'DC9', 'HV5', 'HV11', 'HV12', 'HV13']
    nameB_list = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C9', 'H1', 'H2', 'H3', 'H5', 'H6', 'H7', 'H11', 'H12', 'H13', 'DC8', 'HV8', 'HV9', 'HV10']
    common_atoms_all = ['N3', 'C5', 'C4', 'H3', 'H7', 'H1', 'C2', 'C6', 'C7', 'S1', 'H2', 'H6']
    lineB_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

    test_df = coordinate_swap.find_R2D_D2R_miss(nameB_list, nameA_list, common_atoms_all, lineB_list, 'B2A')
    df = pd.read_csv(f'{input_path}/coord_swap/find_R2D_D2R_miss.csv')

    for index, row in df.iterrows():
        test_row = test_df[test_df['Name'] == row['Name']]
        assert row['Atom Name Number'] == int(test_row['Atom Name Number'].to_list()[0])
        assert row['Element'] == test_row['Element'].to_list()[0]
        assert row['Direction'] == test_row['Direction'].to_list()[0]
        assert row['Swap'] == test_row['Swap'].to_list()[0]
        assert row['File line'] == int(test_row['File line'].to_list()[0])


def test_sep_merge():
    sample_line = ['36B2C', 'N11549', '3.964', '6.464', '6.901', '-0.0888', '-0.6098', '0.8167']
    test_split = coordinate_swap.sep_merge(sample_line)
    assert test_split[0] == '36B2C'
    assert test_split[1] == 'N'
    assert test_split[2] == '11549'

    sample_line = ['36B2C', 'O211557', '3.983', '6.536', '6.608', '-0.2254', '-0.0182', '-0.1860']
    test_split = coordinate_swap.sep_merge(sample_line)
    assert test_split[0] == '36B2C'
    assert test_split[1] == 'O2'
    assert test_split[2] == '11557'


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
