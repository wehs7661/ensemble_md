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
    test_file1 = open(f'{input_path}/coord_swap/input_A.gro', 'r').readlines()
    test_file2 = open(f'{input_path}/coord_swap/input_B.gro', 'r').readlines()
    test_df = coordinate_swap.find_common(test_file1, test_file2, 'D2E', 'E2F')
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
    nameA_list = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C9',
                  'H1', 'H2', 'H3', 'H5', 'H6', 'H7', 'H11', 'H12', 'H13',
                  'DC8', 'HV8', 'HV9', 'HV10']
    nameB_list = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                  'H1', 'H2', 'H3', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13',
                  'DC10', 'HV4', 'HV14', 'HV15', 'HV16']

    common_atoms_all = ['C6', 'H7', 'C4', 'S1', 'H6', 'H11', 'C7', 'H12', 'H2', 'H1',
                        'C9', 'H3', 'C2', 'N3', 'H13', 'C5']
    lineB_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    test_df = coordinate_swap.find_R2D_D2R_miss(nameB_list, nameA_list, common_atoms_all, lineB_list, 'B2A')
    df = pd.read_csv(f'{input_path}/coord_swap/find_R2D_D2R_miss.csv')

    for index, row in df.iterrows():
        test_row = test_df[test_df['Name'] == row['Name']]
        assert row['Atom Name Number'] == int(test_row['Atom Name Number'].to_list()[0])
        assert row['Element'] == test_row['Element'].to_list()[0]
        assert row['Direction'] == test_row['Direction'].to_list()[0]
        assert row['Swap'] == test_row['Swap'].to_list()[0]
        assert row['File line'] == int(test_row['File line'].to_list()[0])


def test_fix_break():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol.gro')
    df_connect = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    df_connect_res = df_connect[df_connect['Resname'] == 'C2D']
    test_fix = coordinate_swap.fix_break(broken_mol, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    fixed_mol = md.load(f'{input_path}/coord_swap/fixed_mol.gro')

    assert (test_fix.xyz == fixed_mol.xyz).all


def test_perform_shift_1D():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap.perform_shift_1D(broken_mol,
                                                                                     [2.74964, 2.74964, 2.74964], [[0, 4]], [])

    broken_pairs = coordinate_swap.check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0


def test_check_break():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol.gro')
    df_connect = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')

    atom_connect_all = df_connect[df_connect['Resname'] == 'C2D']
    mol_top = broken_mol.topology
    resname = 'C2D'
    atom_connect = []
    for i, row in atom_connect_all.iterrows():
        atom_connect.append([row['Connect 1'], row['Connect 2']])

    atom_pairs = []
    for atoms in atom_connect:
        atom_pairs.append(list(mol_top.select(f"resname {resname} and (name {atoms[0]} or name {atoms[1]})")))

    broken_pairs = coordinate_swap.check_break(broken_mol, atom_pairs)

    assert broken_pairs == [[0, 4], [1, 2]] or broken_pairs == [[1, 2], [0, 4]]


def test_get_miss_coord():
    molA_file = f'{input_path}/coord_swap/input_A.gro'
    molB_file = f'{input_path}/coord_swap/input_B.gro'
    nameA = 'D2E'
    nameB = 'E2F'

    connection_map = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')

    molA = md.load(f'{input_path}/coord_swap/input_A.trr', top=molA_file).slice(-1)
    molB = md.load(f'{input_path}/coord_swap/input_B.trr', top=molB_file).slice(-1)

    A_dimensions = coordinate_swap.get_dimensions(open(molA_file, 'r').readlines())
    B_dimensions = coordinate_swap.get_dimensions(open(molB_file, 'r').readlines())
    molA = coordinate_swap.fix_break(molA, nameA, A_dimensions, connection_map[connection_map['Resname'] == nameA])
    molB = coordinate_swap.fix_break(molB, nameB, B_dimensions, connection_map[connection_map['Resname'] == nameB])

    df_no_coords = pd.read_csv(f'{input_path}/coord_swap/find_common.csv')
    df = pd.read_csv(f'{input_path}/coord_swap/df_atom_swap.csv')
    df_no_nan = df.dropna()

    df_test = coordinate_swap.get_miss_coord(molB, molA, nameB, nameA, df_no_coords, 'B2A',
                                             swap_map[(swap_map['Swap A'] == nameB) & (swap_map['Swap B'] == nameA)])
    df_test = coordinate_swap.get_miss_coord(molA, molB, nameA, nameB, df_test, 'A2B',
                                             swap_map[(swap_map['Swap A'] == nameA) & (swap_map['Swap B'] == nameB)])

    for index, row in df_no_nan.iterrows():
        test_row = df_test[df_test['Name'] == row['Name']]
        assert np.isclose(row['X Coordinates'], test_row['X Coordinates'], atol=10**(-6))
        assert np.isclose(row['Y Coordinates'], test_row['Y Coordinates'], atol=10**(-6))
        assert np.isclose(row['Z Coordinates'], test_row['Z Coordinates'], atol=10**(-6))


def test_process_line():
    file = open(f'{input_path}/coord_swap/input_A.gro', 'r').readlines()

    line, prev_line = coordinate_swap.process_line(file, 5)

    assert prev_line == ['1D2E', 'N3', '3', '2.229', '1.274', '2.620', '0.0270', '-0.2197', '0.0105\n']
    assert line == ['1D2E', 'C4', '4', '2.226', '1.138', '2.607', '-0.8557', '0.2885', '0.1035\n']

    line, prev_line = coordinate_swap.process_line(file, 22)
    assert prev_line == ['1D2E', 'HV9', '20', '2.510', '1.423', '2.489', '1.4858', '0.4341', '-3.5063\n']
    assert line == ['1D2E', 'HV10', '21', '2.676', '1.415', '2.541', '-0.1731', '-0.2227', '-0.1934\n']


def test_print_preamble():
    file = open(f'{input_path}/coord_swap/input_A.gro', 'r').readlines()
    temp_file = open('test_print_preamble.gro', 'w')

    coordinate_swap.print_preamble(file, temp_file, 5, 1)
    temp_file.close()

    read_temp_file = open('test_print_preamble.gro', 'r').readlines()
    num_atoms_temp_file = int(read_temp_file[1])
    assert num_atoms_temp_file == 2062
    os.remove('test_print_preamble.gro')


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
