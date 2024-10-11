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
    test_file1 = open(f'{input_path}/coord_swap/sim_A/confout_backup.gro', 'r').readlines()
    test_file2 = open(f'{input_path}/coord_swap/sim_B/confout_backup.gro', 'r').readlines()
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

    test_df = coordinate_swap._find_R2D_D2R_miss(nameB_list, nameA_list, common_atoms_all, lineB_list, 'B2A')
    df = pd.read_csv(f'{input_path}/coord_swap/find_R2D_D2R_miss.csv')

    for index, row in df.iterrows():
        test_row = test_df[test_df['Name'] == row['Name']]
        assert row['Atom Name Number'] == int(test_row['Atom Name Number'].to_list()[0])
        assert row['Element'] == test_row['Element'].to_list()[0]
        assert row['Direction'] == test_row['Direction'].to_list()[0]
        assert row['Swap'] == test_row['Swap'].to_list()[0]
        assert row['File line'] == int(test_row['File line'].to_list()[0])


def test_fix_break():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol_1D.gro')
    df_connect = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    df_connect_res = df_connect[df_connect['Resname'] == 'C2D']
    test_fix = coordinate_swap.fix_break(broken_mol, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    broken_mol_3D = md.load(f'{input_path}/coord_swap/broken_mol_3D.gro')
    test_fix_3D = coordinate_swap.fix_break(broken_mol_3D, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    already_fixed = md.load(f'{input_path}/coord_swap/fixed_mol.gro')
    still_fixed = coordinate_swap.fix_break(already_fixed, 'C2D', [2.74964, 2.74964, 2.74964], df_connect_res)

    fixed_mol = md.load(f'{input_path}/coord_swap/fixed_mol.gro')

    assert (test_fix.xyz == fixed_mol.xyz).all
    assert (test_fix_3D.xyz == fixed_mol.xyz).all
    assert (still_fixed.xyz == fixed_mol.xyz).all


def test_perform_shift():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol_1D.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap._perform_shift(broken_mol, [2.74964, 2.74964, 2.74964], [[0, 4]], [], 1)  # noqa: E501

    broken_pairs = coordinate_swap._check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0

    broken_mol_2D = md.load(f'{input_path}/coord_swap/broken_mol_2D.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap._perform_shift(broken_mol_2D, [2.74964, 2.74964, 2.74964], [[0, 4]], [], 2)  # noqa: E501

    broken_pairs = coordinate_swap._check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0

    broken_mol_3D = md.load(f'{input_path}/coord_swap/broken_mol_3D.gro')

    partial_fix, was_it_fixed, prev_shifted_atoms = coordinate_swap._perform_shift(broken_mol_3D, [2.74964, 2.74964, 2.74964], [[0, 4]], [], 3)  # noqa: E501

    broken_pairs = coordinate_swap._check_break(partial_fix, [[0, 4]])

    assert prev_shifted_atoms == [4]
    assert was_it_fixed is True
    assert len(broken_pairs) == 0


def test_check_break():
    broken_mol = md.load(f'{input_path}/coord_swap/broken_mol_1D.gro')
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

    broken_pairs = coordinate_swap._check_break(broken_mol, atom_pairs)

    assert broken_pairs == [[0, 4], [1, 2]] or broken_pairs == [[1, 2], [0, 4]]


def test_get_miss_coord():
    molA_file = f'{input_path}/coord_swap/sim_A/confout_backup.gro'
    molB_file = f'{input_path}/coord_swap/sim_B/confout_backup.gro'
    nameA = 'D2E'
    nameB = 'E2F'

    connection_map = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')
    swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')

    molA = md.load(f'{input_path}/coord_swap/sim_A/traj.trr', top=molA_file).slice(-1)
    molB = md.load(f'{input_path}/coord_swap/sim_B/traj.trr', top=molB_file).slice(-1)

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
    file = open(f'{input_path}/coord_swap/sample_process.gro', 'r').readlines()

    line, prev_line = coordinate_swap._process_line(file, 5)

    assert prev_line == ['19GLU', 'H2', '3', '1.800', '4.155', '4.152', '2.0821', '0.4011', '-1.2490\n']
    assert line == ['19GLU', 'H3', '4', '1.833', '4.306', '4.113', '-0.3604', '0.3093', '-1.3761\n']

    line, prev_line = coordinate_swap._process_line(file, 15429)
    assert prev_line == ['4487SOL', 'HW1', '15427', '5.528', '0.500', '4.439', '1.2407', '1.5381', '-0.1116\n'] #15429
    assert line == ['4487SOL', 'HW2', '15428', '5.477', '0.613', '4.526', '-0.4379', '1.1843', '-0.6399\n'] #15430


def test_print_preamble():
    file = open(f'{input_path}/coord_swap/sim_A/confout_backup.gro', 'r').readlines()
    temp_file = open('test_print_preamble.gro', 'w')

    coordinate_swap.print_preamble(file, temp_file, 5, 1)
    temp_file.close()

    read_temp_file = open('test_print_preamble.gro', 'r').readlines()
    num_atoms_temp_file = int(read_temp_file[1])
    assert num_atoms_temp_file == 2062
    os.remove('test_print_preamble.gro')


def test_write_line():
    test_file = open('test_write_line.gro', 'w')

    line_merged = '   36F2G    C1011574   3.917   6.393   5.463  0.2985 -0.1406  0.4882/n'
    line = ['36F2G', 'C10', '11574', '3.917', '6.393', '5.463', '0.2985', '-0.1406', '0.4882/n']
    new_coord = [3.9165084, 6.3927655, 5.4633074]
    vel = ['0.000', '0.000', '0.000\n']
    atom_num = 11574

    coordinate_swap.write_line(test_file, line_merged, line, atom_num, vel, new_coord, 36, 'E2F')

    line_merged = '  812SOL     OW12270   5.440   0.656   8.311  0.4628 -0.0392  0.2554/n'
    line = ['812SOL', 'OW', '12270', '5.440', '0.656', '8.311', '0.4628', '-0.0392', '0.2554/n']
    new_coord = [5.4400544, 0.6561325, 8.3108530]
    atom_num = 12264
    coordinate_swap.write_line(test_file, line_merged, line, atom_num, vel, new_coord)
    line_merged = '   2.74964   2.74964   2.74964\n'
    line = ['2.74964', '2.74964', '2.74964\n']
    coordinate_swap.write_line(test_file, line_merged, line, atom_num, vel, new_coord)
    test_file.close()

    reopen_test = open('test_write_line.gro', 'r').readlines()

    assert reopen_test[0] == '   36E2F    C1011574   3.9165084   6.3927655   5.4633074   0.000   0.000   0.000\n'
    assert reopen_test[1] == '  812SOL     OW12264   5.4400544   0.6561325   8.3108530   0.000   0.000   0.000\n'
    assert reopen_test[2] == '   2.74964   2.74964   2.74964\n'
    os.remove('test_write_line.gro')


def test_identify_res():
    swap_map = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')

    for file_name, real_name in zip(['A-B', 'B-C', 'C-D', 'D-E', 'E-F'], ['A2B', 'B2C', 'C2D', 'D2E', 'E2F']):
        mol = md.load(f'{input_path}/coord_swap/{file_name}.gro')
        residue_options = swap_map['Swap A'].to_list() + swap_map['Swap B'].to_list()
        name = coordinate_swap.identify_res(mol.topology, residue_options)
        assert name == real_name


def test_add_atom():
    df = pd.read_csv(f'{input_path}/coord_swap/df_atom_swap.csv')
    temp_file = open('test_add_atom.gro', 'w')

    coordinate_swap._add_atom(temp_file, 1, 'D2E', df[(df['Name'] == 'H5') & (df['Swap'] == 'A2B')], ['0.000', '0.000', '0.000\n'], 12)  # noqa: E501
    coordinate_swap._add_atom(temp_file, 1, 'E2F', df[(df['Name'] == 'DC10') & (df['Swap'] == 'B2A')], ['0.000', '0.000', '0.000\n'], 21)  # noqa: E501
    temp_file.close()

    read_temp_file = open('test_add_atom.gro', 'r').readlines()
    assert read_temp_file[0] == '    1D2E     H5   12   0.0926050   1.6340195   0.3355029   0.000   0.000   0.000\n'
    assert read_temp_file[1] == '    1E2F   DC10   21   2.6200285   1.4039259   2.7885396   0.000   0.000   0.000\n'
    os.remove('test_add_atom.gro')


def test_dummy_real_swap():
    test_file = open('test_dummy_real_swap.gro', 'w')
    df = pd.read_csv(f'{input_path}/coord_swap/df_atom_swap.csv')
    orig_coords = np.zeros((22, 3))
    orig_coords[17] = [2.5837841, 1.4738766, 2.5511920]

    coordinate_swap._dummy_real_swap(test_file, 1, 'E2F', df[df['Name'] == 'DC8'], ['0.000', '0.000', '0.000\n'], 8, orig_coords, 'C8')  # noqa: E501
    test_file.close()

    reopen_test_file = open('test_dummy_real_swap.gro', 'r').readlines()
    assert reopen_test_file[0] == '    1E2F     C8    8   2.5837841   1.4738766   2.5511920   0.000   0.000   0.000\n'
    os.remove('test_dummy_real_swap.gro')


def test_sep_merge():
    sample_line = ['36B2C', 'N11549', '3.964', '6.464', '6.901', '-0.0888', '-0.6098', '0.8167']
    test_split = coordinate_swap._sep_merge(sample_line)
    assert test_split[0] == '36B2C'
    assert test_split[1] == 'N'
    assert test_split[2] == '11549'

    sample_line = ['36B2C', 'O211557', '3.983', '6.536', '6.608', '-0.2254', '-0.0182', '-0.1860']
    test_split = coordinate_swap._sep_merge(sample_line)
    assert test_split[0] == '36B2C'
    assert test_split[1] == 'O2'
    assert test_split[2] == '11557'


def test_rotate_point_around_axis():
    initial_point = np.array([0.16, 0.19, -0.05])
    vertex = np.array([0, 0, 0])
    axis = np.array([0.15, 0.82, 0.14])
    angle = 0.13
    rotated_point = [0.1693233, 0.18548463, -0.0335421]
    assert (coordinate_swap._rotate_point_around_axis(initial_point, vertex, axis, angle) == rotated_point).all


def test_find_rotation_angle():
    initial_point = np.array([0.16, 0.19, -0.05])
    vertex = np.array([0, 0, 0])
    axis = np.array([0.15, 0.82, 0.14])
    rotated_point = [0.1693233, 0.18548463, -0.0335421]
    angle = 0.13
    test_angle = coordinate_swap._find_rotation_angle(initial_point, vertex, rotated_point, axis)
    assert np.isclose(angle, test_angle, 10**(-5))


def test_add_or_swap():
    test_file = open('test_add_or_swap.gro', 'w')
    df = pd.read_csv(f'{input_path}/coord_swap/df_atom_swap.csv')
    orig_coords = np.zeros((21, 3))
    orig_coords[18] = [2.5970387, 1.5708300, 2.5017865]
    skip_line = []

    skip_line = coordinate_swap._add_or_swap(df[df['Name'] == 'HV4'], test_file, 1, 'E2F', ['0.000', '0.000', '0.000\n'], 22, orig_coords, skip_line, 'HV4')  # noqa: E501
    skip_line = coordinate_swap._add_or_swap(df[df['Name'] == 'HV8'], test_file, 1, 'E2F', ['0.000', '0.000', '0.000\n'], 15, orig_coords, skip_line, 'H8')  # noqa: E501
    test_file.close()

    reopen_test_file = open('test_add_or_swap.gro', 'r').readlines()
    assert reopen_test_file[0] == '    1E2F    HV4   22   2.3651702   1.4678032   2.8239074   0.000   0.000   0.000\n'
    assert reopen_test_file[1] == '    1E2F     H8   15   2.5970387   1.5708300   2.5017865   0.000   0.000   0.000\n'
    os.remove('test_add_or_swap.gro')

    assert skip_line == [20]


def test_swap_name():
    df_top = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')

    name_list = ['C2', 'C5', 'DC7', 'HV5']
    flip_name_list = ['C2', 'C5', 'C7', 'H5']
    test_name_list = coordinate_swap._swap_name(name_list, 'B2C', df_top)
    test_flip_name_list = coordinate_swap._swap_name(flip_name_list, 'A2B', df_top)

    assert test_name_list == flip_name_list
    assert test_flip_name_list == name_list


def test_get_names():
    top_files = ['A-B.itp', 'B-C.itp', 'C-D.itp', 'D-E.itp', 'E-F.itp']

    start_lines = [26, 29, 33, 32, 36]
    names = [['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'H1', 'H2', 'H3', 'H4', 'H17', 'DC7', 'HV5', 'HV6', 'HV7'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'DC8', 'HV8', 'HV9', 'HV10'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'H1', 'H2', 'H3', 'H4', 'H6', 'H7', 'H8', 'H9', 'H10', 'DC9', 'HV5', 'HV11', 'HV12', 'HV13'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C9', 'H1', 'H2', 'H3', 'H5', 'H6', 'H7', 'H11', 'H12', 'H13', 'DC8', 'HV8', 'HV9', 'HV10'], ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 'H2', 'H3', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'DC10', 'HV4', 'HV14', 'HV15', 'HV16']]  # noqa: E501

    lambda_states = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
                     [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0]]
    for i, top_file in enumerate(top_files):
        top = open(f'{input_path}/coord_swap/{top_file}', 'r').readlines()
        test_start_line, test_names, test_lambda_states = coordinate_swap.get_names(top)
        assert test_start_line == start_lines[i]
        assert test_names == names[i]
        assert test_lambda_states == lambda_states[i]


def test_determine_connection():
    cmpr_df = pd.read_csv(f'{input_path}/coord_swap/residue_swap_map.csv')
    df_top = pd.read_csv(f'{input_path}/coord_swap/residue_connect.csv')

    A2B_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'H1', 'H2', 'H3', 'H4', 'H17', 'DC7', 'HV5', 'HV6', 'HV7']
    B2C_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'DC8', 'HV8', 'HV9', 'HV10']  # noqa: E501
    D2E_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C9', 'H1', 'H2', 'H3', 'H5', 'H6', 'H7', 'H11', 'H12', 'H13', 'DC8', 'HV8', 'HV9', 'HV10']  # noqa: E501
    E2F_names = ['S1', 'C2', 'N3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'H1', 'H2', 'H3', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'DC10', 'HV4', 'HV14', 'HV15', 'HV16']  # noqa: E501

    A_only = [x for x in A2B_names if x not in B2C_names]
    B_only = [x for x in B2C_names if x not in A2B_names]
    test_df = coordinate_swap.determine_connection(A_only, B_only, 'A2B', 'B2C', df_top, 1)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap A'] == 'A2B') & (cmpr_df['Swap B'] == 'B2C')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]

    test_df = coordinate_swap.determine_connection(B_only, A_only, 'B2C', 'A2B', df_top, 0)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap B'] == 'A2B') & (cmpr_df['Swap A'] == 'B2C')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]

    A_only = [x for x in D2E_names if x not in E2F_names]
    B_only = [x for x in E2F_names if x not in D2E_names]
    test_df = coordinate_swap.determine_connection(A_only, B_only, 'D2E', 'E2F', df_top, 1)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap A'] == 'D2E') & (cmpr_df['Swap B'] == 'E2F')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]

    test_df = coordinate_swap.determine_connection(B_only, A_only, 'E2F', 'D2E', df_top, 0)
    select_cmpr_df = cmpr_df[(cmpr_df['Swap A'] == 'E2F') & (cmpr_df['Swap B'] == 'D2E')]
    for col in ['Anchor Atom Name A', 'Anchor Atom Name B', 'Alignment Atom A', 'Alignment Atom B', 'Angle Atom A', 'Angle Atom B', 'Missing Atom Name']:  # noqa: E501
        assert test_df[col].to_list()[0] == select_cmpr_df[col].to_list()[0]
