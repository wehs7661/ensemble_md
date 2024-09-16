import mdtraj as md
import re
import numpy as np
import os
from itertools import product
import pandas as pd
import copy

def get_dimensions(file):
    """
    Determine the dimensions of the cubic box based on the input GRO file
    
    Parameters
    ----------
    file : List of strings 
        containing all of the lines in the input gro file

    Returns
    -------
    [x, y, z] : List of floats 
        X, Y, and Z coordinates for the periodic box
    """
    box_raw = file[-1].split(' ')
    while '' in box_raw:
        box_raw.remove('')
    
    return [float(box_raw[0]), float(box_raw[1]), float(box_raw[2])]

def deter_common(molA_file, molB_file,  nameA : str, nameB : str):
    """
    Determine the atoms which are common, which are switched between dummy and real atoms, and which are unique between the two molecule

    Parameters
    ----------
    molA_file : List of strings
        Lines of the GRO file for molecule A
    molB_file : List of strings
        Lines of the GRO file for molecule B
    nameA : str
        The name of the residue being altered in molecule A
    nameB : str
        The name of the residue being altered in molecule B
    
    Returns
    -------
    df : pandas DataFrame
        Containing the atoms which are not the same between the two molecules and how they are changed
    """
    #Gather atom names from each file
    nameA_list, lineA_list, nameB_list, lineB_list = [], [], [], []
    for l, line in enumerate(molA_file):
        split_line = line.split(' ')
        while("" in split_line): split_line.remove("")
        if len(line[1]) > 5:
            split_line = sep_merge(split_line)
        if nameA in split_line[0]:
            nameA_list.append(split_line[1])
            lineA_list.append(l)

    for l, line in enumerate(molB_file):
        split_line = line.split(' ')
        while("" in split_line): split_line.remove("")
        if len(line[1]) > 5:
            split_line = sep_merge(split_line)
        if nameB in split_line[0]:
            nameB_list.append(split_line[1])
            lineB_list.append(l)

    #Determine the atom names present in both molecules
    common_atoms_all = list(set(nameA_list) & set(nameB_list))

    #Determine the swaps for each transformation
    df_A2B = deter_R2D_D2R_miss(nameA_list, nameB_list, common_atoms_all, lineA_list, 'A2B')
    df_B2A = deter_R2D_D2R_miss(nameB_list, nameA_list, common_atoms_all, lineB_list, 'B2A')

    #Add D2R
    df = pd.concat([df_A2B, df_B2A])
    df.reset_index(inplace=True)
    return df

def deter_R2D_D2R_miss(name_list, name_other_list, common_atoms, line_list, swap):
    """
    Determine which atoms are swapped between dummy and real states and which are missing

    Parameters
    ----------
    name_list : str
        List of all the atom names for the atoms in the molecule of interest
    name_other_list : str
        List of all the atom names for the atoms that the molecule of interest is swapping with
    common_atoms : str
        List of all the atom names for atoms known to be common between the two molecules
    Line_list : int
        List of the line in the file which corresponds to each atom name in the molecule of interest
    swap : str
        The direction of the swap based on the molecules identified as A and B

    Returns
    -------
    df : pandas DataFrame
        For each atom that is not in the common list the dataframe gives the name of the atom, the atom number, the atom element, whether the atom is switchin between dummy and real or missing, the swap direction (same as input), whether the final atom type is real or dummy 
    """
    #Determine atoms unique to either molecule
    names, an_num, elements, directions, swaps, lines, final_type = [], [], [], [], [], [], []
    for a, atom in enumerate(name_list):
        element = atom.strip('0123456789')
        real_element = element.strip('DV')
        num = atom.strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if atom not in common_atoms:
            names.append(atom)
            an_num.append(num)
            swaps.append(swap)
            lines.append(line_list[a])
            if 'C' in element:
                elements.append('C')
            else:
                elements.append('H')
            if f'D{atom}' in name_other_list or f'{element}V{num}' in name_other_list:
                directions.append('R2D')
                final_type.append('dummy')
            elif f'{real_element}{num}' in name_other_list:
                directions.append('D2R')
                final_type.append('real')
            else:
                directions.append('miss')
                if 'V' in atom or 'DC' in atom:
                    final_type.append('dummy')
                else:
                    final_type.append('real')
    return pd.DataFrame({'Name': names, 'Atom Name Number': an_num, 'Element': elements, 'Direction': directions, 'Swap': swaps, 'File line': lines, 'Final Type': final_type})

def fix_break(mol, resname, box_dimensions, atom_connect_all):
    """
    Determine if there are any breaks across periodic boundary conditions in the residue of interest and shift the molecule across those boudaries to make it whole

    Parameters
    ----------
    mol : mdtraj Trajectory object
        Trajectory object for the molecule or interest
    resname : str
        Name of the residue of interest for coordinate swapping
    box_dimensions : list
        Dimensions of the periodic box for the system of interest
    atom_connect_all : pd.DataFrame
        DataFrame which contains the name of all atoms which are connected to one another in the residue of interest
    
    Returns
    -------
    mol : mdtraj Trajecotry object
        Trajectory with all breaks in the necessary portions of the molecule fixed

    """
    mol_top = mol.topology
    
    atom_connect = []
    for i, row in atom_connect_all.iterrows():
        atom_connect.append([row['Connect 1'], row['Connect 2']])
    
    atom_pairs = []
    for atoms in atom_connect:
        atom_pairs.append(list(mol_top.select(f"resname {resname} and (name {atoms[0]} or name {atoms[1]})")))
    
    #Find any broken bonds
    broken_pairs = check_break(mol, atom_pairs)

    if len(broken_pairs) == 0:
        print('No breaks found')
        return mol
    else:
        print('Fixing break')

    iter = 0 #Keep track of while loop iterations
    shift_atom = []
    while len(broken_pairs) > 0:
        #Prevent infinite loop
        if iter > len(atom_pairs):
            raise Exception('Break could not be fixed') 
        iter += 1
        
        #Fix this break
        mol, Fixed, shift_atom = perform_shift_1D(mol, box_dimensions, broken_pairs, atom_pairs, shift_atom)
        if Fixed:
            broken_pairs = check_break(mol, atom_pairs)
            continue
        else:
            mol, Fixed = perform_shift_2D(mol, box_dimensions, broken_pairs, shift_atom)
            if Fixed:
                broken_pairs = check_break(mol, atom_pairs)
                continue
            else:
                mol, Fixed = perform_shift_3D(mol, box_dimensions, broken_pairs, shift_atom)
                if Fixed:
                    broken_pairs = check_break(mol, atom_pairs)
                    continue
                else:
                    raise Exception('Break could not be fixed')  
    
    return mol

def perform_shift_1D(mol, box_dimensions, broken_pairs_init, prev_shift_atom):
    """
    Shift across the periodic boundaries in 1D

    Parameters
    ----------
    mol : mdtraj Trajectory object
        Trajecotry with the original coordinates prior to the fhist
    box_dimensions : list of floats
        Dimensions of the periodic boundary box
    broken_pairs_init : int
        which pairs of atoms were found to be broken
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done 

    Returns
    -------
    mol : mdtraj Trajectory object
        Trajectory with the new coordintes
    Fixed : bool
        Was the break actually fixed 
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done
    """
    atom_pair = broken_pairs_init[0]
    broken_atom = atom_pair[1]
    if broken_atom in prev_shift_atom:
        broken_atom = atom_pair[0]
    Fixed = False
    for x in range(3): #Loop through x, y, and z
        for shift_dir in [1, -1]:
            mol.xyz[0,broken_atom,x] = mol.xyz[0,broken_atom,x] + (shift_dir[0] * box_dimensions[x]) #positive shift
            dist_check = md.compute_distances(mol, atom_pairs=[atom_pair], periodic=False)
            if dist_check > 0.2: #Didn't work so reverse and try again
                mol.xyz[0,broken_atom,x] = mol.xyz[0,broken_atom,x] - (shift_dir[0] * box_dimensions[x])
            else: #Yay fixed break
                Fixed = True
                break
        if Fixed:
            break
    if Fixed:
        prev_shift_atom.append(broken_atom)
    return mol, Fixed, prev_shift_atom

def perform_shift_2D(mol, box_dimensions, broken_pairs_init, prev_shift_atom):
    """
    Shift across the periodic boundaries in 2D

    Parameters
    ----------
    mol : mdtraj Trajectory object
        Trajecotry with the original coordinates prior to the fhist
    box_dimensions : list of floats
        Dimensions of the periodic boundary box
    broken_pairs_init : int
        which pairs of atoms were found to be broken
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done 

    Returns
    -------
    mol : mdtraj Trajectory object
        Trajectory with the new coordintes
    Fixed : bool
        Was the break actually fixed 
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done
    """
    atom_pair = broken_pairs_init[0]
    broken_atom = atom_pair[1]
    if broken_atom in prev_shift_atom:
        broken_atom = atom_pair[0]
    Fixed = False
    shift_combos = product([0, 1, 2], [0, 1, 2])
    for pair in shift_combos: #Loop through x, y, and z
        for shift_dir in product([1, -1], [1,-1]): #Try all combos of shift direction
            x, y = pair
            mol.xyz[0,broken_atom,x] = mol.xyz[0,broken_atom,x] + (shift_dir[0] * box_dimensions[x]) #positive shift
            mol.xyz[0,broken_atom,y] = mol.xyz[0,broken_atom,y] + (shift_dir[1] * box_dimensions[y])
            dist_check = md.compute_distances(mol, atom_pairs=[atom_pair], periodic=False)
            if dist_check > 0.2: #Didn't work so reverse and try again
                mol.xyz[0,broken_atom,x] = mol.xyz[0,broken_atom,x] - (shift_dir[0] * box_dimensions[x])
                mol.xyz[0,broken_atom,y] = mol.xyz[0,broken_atom,y] - (shift_dir[1] * box_dimensions[y])
            else: #Yay fixed break
                Fixed = True
                break
        if Fixed:
            break
    if Fixed:
        prev_shift_atom.append(broken_atom)
    return mol, Fixed

def perform_shift_3D(mol, box_dimensions, broken_pairs_init, prev_shift_atom):
    """
    Shift across the periodic boundaries in 3D

    Parameters
    ----------
    mol : mdtraj Trajectory object
        Trajecotry with the original coordinates prior to the fhist
    box_dimensions : list of floats
        Dimensions of the periodic boundary box
    broken_pairs_init : int
        which pairs of atoms were found to be broken
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done 

    Returns
    -------
    mol : mdtraj Trajectory object
        Trajectory with the new coordintes
    Fixed : bool
        Was the break actually fixed 
    prev_shift_atom : int
        Which atoms have already been shifted so we don't undo what we've done
    """
    atom_pair = broken_pairs_init[0]
    broken_atom = atom_pair[1]
    if broken_atom in prev_shift_atom:
        broken_atom = atom_pair[0]
    Fixed = False
    for shift_dir in product([1, -1], [1,-1], [1, -1]): #Try all combos of shift direction
        mol.xyz[0,broken_atom,0] = mol.xyz[0,broken_atom,0] + (shift_dir[0] * box_dimensions[0]) 
        mol.xyz[0,broken_atom,1] = mol.xyz[0,broken_atom,1] + (shift_dir[1] * box_dimensions[1])
        mol.xyz[0,broken_atom,2] = mol.xyz[0,broken_atom,2] + (shift_dir[2] * box_dimensions[2])
        dist_check = md.compute_distances(mol, atom_pairs=[atom_pair], periodic=False)
        if dist_check > 0.2: #Didn't work so reverse and try again
            mol.xyz[0,broken_atom,0] = mol.xyz[0,broken_atom,0] - (shift_dir[0] * box_dimensions[0]) 
            mol.xyz[0,broken_atom,1] = mol.xyz[0,broken_atom,1] - (shift_dir[1] * box_dimensions[1])
            mol.xyz[0,broken_atom,2] = mol.xyz[0,broken_atom,2] - (shift_dir[2] * box_dimensions[2])
        else: #Yay fixed break
            Fixed = True
            break
    if Fixed:
        prev_shift_atom.append(broken_atom)
    return mol, Fixed

def check_break(mol, atom_pairs):
    """
    Determine whether a break is present between the atom pairs of interest

    Parameters
    ----------
    mol : mdtraj Trajectory object
        Trajectory to examine for breaks
    atom_pairs : int
        The atom number pairs which should be connected in the residue of interest
    
    Returns
    -------
    broken_pairs : int
        which pairs of atoms have breaks across the periodic boundary
    """
    dist = md.compute_distances(mol, atom_pairs=atom_pairs, periodic=False)
    broken_pairs = []
    for i, d in enumerate(dist[0]):
        if d > 0.2:
            broken_pairs.append(atom_pairs[i])
    return broken_pairs

def get_miss_coord(mol_align, mol_ref, name_align, name_ref, df_atom_swap, dir, df_swap):
    """
    Get coordinates for the missing atoms after the conformational swap

    Parameters
    ----------
    mol_align : mdtraj Trajectory object
        trajectory for the molecule being aligned
    mol_ref : mdtraj Trajectory object
        trajectory for the reference molecule
    name_align : str
        resname for the molecule which is being aligned
    name_ref : str
        resname for the molecule which has the reference coordinates
    df_atom_swap : pandas DataFrame
        For each atom that is not in the common list the dataframe gives the name of the atom, the atom number, the atom element, whether the atom is switchin between dummy and real or missing, the swap direction (same as input), whether the final atom type is real or dummy 
    dir : str
        swapping direction
    df_swap : pandas DataFrame
        For each missing atoms 
    
    Returns
    -------
    df_atom_swap : pandas DataFrame
        Same dataframe as the input, but with coordinates for the missing atoms
    """
    #Create a new column for coordinates if one does not exist
    if 'X Coordinates' not in df_atom_swap.columns:
        df_atom_swap['X Coordinates'] = np.NaN
        df_atom_swap['Y Coordinates'] = np.NaN
        df_atom_swap['Z Coordinates'] = np.NaN
    
    if len(df_swap.index) == 0:
        return df_atom_swap
    
    for i, row in df_swap.iterrows():
        conn_align = [row['Anchor Atom Name A'], row['Alignment Atom A']]
        conn_ref = [row['Anchor Atom Name B'], row['Alignment Atom B']]
        miss_names_select = row['Missing Atom Name'].split(' ')
        geom_fix_select = [miss_names_select[0], row['Anchor Atom Name A'], row['Angle Atom A']]
        #Limit to region of interest
        align_select = []
        for a in conn_align + miss_names_select + geom_fix_select:
            atom = mol_align.topology.select(f'resname {name_align} and name {a}')[0]
            if atom not in align_select:
                align_select.append(atom)
        align_select.sort()
        ref_select = mol_ref.topology.select(f"resname {name_ref} and (name {conn_ref[0]} or name {conn_ref[1]} or name {row['Angle Atom B']})")
        
        mol_ref_select = copy.deepcopy(mol_ref.atom_slice(ref_select))
        mol_align_select = copy.deepcopy(mol_align.atom_slice(align_select))

        #Select atoms in connection bond
        conn0_align = mol_align_select.topology.select(f'resname {name_align} and name {conn_align[0]}')[0]
        conn1_align = mol_align_select.topology.select(f'resname {name_align} and name {conn_align[1]}')[0]
        conn0_ref = mol_ref_select.topology.select(f'resname {name_ref} and name {conn_ref[0]}')[0]
        conn1_ref = mol_ref_select.topology.select(f'resname {name_ref} and name {conn_ref[1]}')[0]

        #Step 1: Perform translation to align point 1
        shift_vec = mol_ref_select.xyz[0,conn0_ref,:] - mol_align_select.xyz[0,conn0_align,:]
        for a in range(mol_align_select.n_atoms):
            mol_align_select.xyz[0,a,:] = mol_align_select.xyz[0,a,:] + shift_vec

        #Step 2: Perform rotational motion to line up point 2
        ref_vec = mol_ref_select.xyz[0,conn1_ref,:] - mol_ref_select.xyz[0,conn0_ref,:] #defing 0-1 vector in ref
        align_vec = mol_align_select.xyz[0,conn1_align,:] - mol_ref_select.xyz[0,conn0_ref,:] #defing 0-1 vector in align
        axis_rot = np.cross(ref_vec/np.linalg.norm(ref_vec), (align_vec/np.linalg.norm(align_vec))) #Perpendicular vector to ref and align vectors
        theta = find_rotation_angle(mol_align_select.xyz[0,conn1_align,:], mol_ref_select.xyz[0,conn0_ref,:], mol_ref_select.xyz[0,conn1_ref,:], axis_rot)

        for a in range(mol_align_select.n_atoms):
            if a != conn0_align:
                mol_align_select.xyz[0,a,:] = rotate_point_around_axis(mol_align_select.xyz[0,a,:], mol_ref_select.xyz[0,conn0_ref,:], axis_rot, theta)

        #Step 3: Rotate around 0-1 bond to get the correct angle for connecting atom
        angle_atoms = [mol_align.topology.select(f'resname {name_align} and name {geom_fix_select[0]}')[0], 
                       mol_align.topology.select(f'resname {name_align} and name {geom_fix_select[1]}')[0], 
                       mol_align.topology.select(f'resname {name_align} and name {geom_fix_select[2]}')[0]]
        internal_angle = compute_angle([mol_align.xyz[0,angle_atoms[0],:], mol_align.xyz[0,angle_atoms[1],:], mol_align.xyz[0,angle_atoms[2],:]])
        change_atoms = mol_align_select.topology.select(f'resname {name_align} and name {geom_fix_select[0]}')
        axis_rot = mol_align_select.xyz[0,conn0_align,:] -  mol_align_select.xyz[0,conn1_align,:]
        axis_rot = axis_rot / np.linalg.norm(axis_rot)

        #Determine angle of rotation to get the right internal angle
        min_dev_angle = 3
        init_coords = mol_align_select.xyz[0,change_atoms,:][0].copy()
        constant_coords = [mol_align_select.xyz[0,mol_align_select.topology.select(f'resname {name_align} and name {geom_fix_select[1]}')[0],:],
                           mol_ref_select.xyz[0,mol_ref_select.topology.select(f"resname {name_ref} and name {row['Angle Atom B']}")[0],:]]
        for theta in np.linspace(0, 2*np.pi, num=10):
            new_coors = np.zeros((3,3))
            new_coors[0,:] = rotate_point_around_axis(init_coords, mol_ref_select.xyz[0,conn0_ref,:], axis_rot, theta)
            new_coors[1:,:] = constant_coords
            iangle = compute_angle(new_coors)
            if abs(iangle - internal_angle) < min_dev_angle:
                theta_min = theta
                min_dev_angle = abs(iangle - internal_angle)
                if min_dev_angle < 0.05:
                    break
        #Perform the rotation
        for a in range(mol_align_select.n_atoms):
            if a != conn0_align:
                mol_align_select.xyz[0,a,:] = rotate_point_around_axis(mol_align_select.xyz[0,a,:], mol_ref_select.xyz[0,conn0_ref,:], axis_rot, theta_min)

        #Add coordinates to df
        for r in range(len(df_atom_swap.index)):
            if df_atom_swap.iloc[r]['Direction'] == 'miss' and df_atom_swap.iloc[r]['Swap'] == dir:
                for name in miss_names_select:
                    if df_atom_swap.iloc[r]['Name'] == name:
                        a = mol_align_select.topology.select(f'name {name}')
                        df_atom_swap.at[r, 'X Coordinates'] = mol_align_select.xyz[0,a,0]
                        df_atom_swap.at[r, 'Y Coordinates'] = mol_align_select.xyz[0,a,1]
                        df_atom_swap.at[r, 'Z Coordinates'] = mol_align_select.xyz[0,a,2]
                        continue
    
    return df_atom_swap

def process_line(mol_file, i):
    """
    Seperate the line by spaces and correct for when the atom number merges the atom name

    Parameters
    ----------
    mol_file : list of str
        Reading the lines of the input gro file
    i : int
        The iterator for the line in the file
    
    Returns
    -------
    line : str
        Line i in the file
    prev_line : str
        Line i-1 in the file
    """
    #Load previous line to determine identity of previous atom
    prev_line = mol_file[i-1].split(' ')
    while("" in prev_line): prev_line.remove("")
    
    #Load current line in file and seperate
    line = mol_file[i].split(' ')
    while("" in line): line.remove("")
    
    #Control for atom name and atom number merging
    if len(line[1]) > 5:
        line = sep_merge(line)
    if i > 2 and len(prev_line[1]) > 5:
        prev_line = sep_merge(prev_line)

    return line, prev_line

def print_preamble(old_file, new_file, num_add_names, num_rm_names):
    """
    Print the preamble of the gro file before the atom specifications begin

    Parameters
    ----------
    old_file : list of str

    new_file : list of str

    num_add_names : int
        Number of atoms to be added to the pre-swap molecule
    num_rm_names : int
        Number of atoms to be removed to the post-swap molecule

    Returns
    -------
    line_start : int
        The line number to start when you continue to read the file
    """
    for i in range(len(old_file)):
        if not re.match(r'[ \t]', old_file[i]) and not re.match('^[0-9]+$', old_file[i]):
            new_file.write(old_file[i])
        else:
            num_atom = int(float(old_file[i]) + num_add_names - num_rm_names)
            new_file.write(f'{num_atom}\n')
            line_start = i + 1
            break
    return line_start

def write_line(mol_new, raw_line, line, atom_num, vel, coor, resnum=None, nameB=None):
    """
    Write a line in the file in which some parameter needs changed and maintain spacing

    Parameters
    ----------
    mol_new : File object
        The new file that is being written to
    raw_line : str
        The unseperated and unchanged line
    Line : List of str

    atom_num : int
        The new atom number to be assigned to the atom in this line
    vel : list of float
        The velocities to be assigned to the atom in this line
    coor : list of float
        The coordinates for the atom in this line
    resnun : int
        The new residue number if it changes else leave deafult of None
    nameB : str
        The new residue name if if changes from what was in the previous file else leave deafult of None

    Returns
    -------
        None
    """
    coor = ["{:.7f}".format(coor[0]), "{:.7f}".format(coor[1]), "{:.7f}".format(coor[2])]
    if len(line) == 9:
        if nameB == None:
            mol_new.write(line[0].rjust(8, ' ') + line[1].rjust(7,' ') + str(atom_num).rjust(5, ' ') + str(coor[0]).rjust(12, ' ') + str(coor[1]).rjust(12, ' ') + str(coor[2]).rjust(12, ' ') + 
                          vel[0].rjust(8, ' ') + vel[1].rjust(8, ' ') + vel[2].rjust(9, ' '))
        else:
            mol_new.write(f'{resnum}{nameB}'.rjust(8, ' ') + line[1].rjust(7,' ') + str(atom_num).rjust(5, ' ') + str(coor[0]).rjust(12, ' ') + str(coor[1]).rjust(12, ' ')  + str(coor[2]).rjust(12, ' ') +
                          vel[0].rjust(8, ' ') + vel[1].rjust(8, ' ') + vel[2].rjust(9, ' '))
    elif len(line) == 8: #Atom name and number blend together because of number size so they need seperated
        if 'HW1' in line[1]:
            atom_name = 'HW1'
        elif 'HW2' in line[1]:
            atom_name = 'HW2'
        else:
            atom_name = re.sub(r'[0-9]', '', line[1])

        if nameB == None:
            mol_new.write(line[0].rjust(8, ' ') + atom_name.rjust(7,' ') + str(atom_num).rjust(5, ' ') + str(coor[0]).rjust(12, ' ') + str(coor[1]).rjust(12, ' ')  + str(coor[2]).rjust(12, ' ') +
                        vel[0].rjust(8, ' ') + vel[1].rjust(8, ' ') + vel[2].rjust(9, ' '))
        else:
            mol_new.write(f'{resnum}{nameB}'.rjust(8, ' ') + atom_name.rjust(7,' ') + str(atom_num).rjust(5, ' ') + str(coor[0]).rjust(12, ' ') + str(coor[1]).rjust(12, ' ')  + str(coor[2]).rjust(12, ' ') +
                        vel[0].rjust(8, ' ') + vel[1].rjust(8, ' ') + vel[2].rjust(9, ' '))
    else:
        mol_new.write(raw_line)

def deter_res(mol_top, resname_options):
    """
    Determine which of the potential residues of interest are in this molecule

    Parameters
    ----------
    mol_top : mdtraj topology
        The molecule topology
    resname_options : list of str
        The potential residue names which may be in the moleucle

    Returns
    -------
    resname : str
        The name of the residue of interest which is in the molecule
    """
    for name in resname_options:
        if len(mol_top.select(f"resname {name}")) != 0:
            resname = name
            break
    return resname

def add_atom(mol_new, resnum, resname, df, vel, atom_num):
    """
    Add a new atom to the GRO file

    Parameters
    ----------
    mol_new : file
        The temporary file for the new GRO
    resnum : int
        The residue number of the atom being added
    resname : str
        The name of the residue which the atom being added belongs to
    df : pandas.DataFrame
        The master dataframe which contains the coordinates for the atom being added
    vel : list of float
        The velocity to assign to the atom being added
    atom_num : int
        The atom number to assign to the atom being added

    Returns
    -------
    atom_num : int
        The atom number for the next atom in the file
    """
    name_new = df['Name'].to_list()[0]
    #Reformat coordiantes
    x_init = df['X Coordinates'].to_list()
    y_init = df['Y Coordinates'].to_list()
    z_init = df['Z Coordinates'].to_list()
    x_temp = np.round(x_init[0], decimals=7)
    x = format(x_temp, '.7f')
    y_temp = np.round(y_init[0], decimals=7)
    y = format(y_temp, '.7f')
    z_temp = np.round(z_init[0], decimals=7)
    z = format(z_temp, '.7f')
    #Write line for new atom
    mol_new.write(f'{resnum}{resname}'.rjust(8, ' ') + name_new.rjust(7, ' ') + str(atom_num).rjust(5, ' ') + str(x).rjust(12, ' ')  + str(y).rjust(12, ' ') + 
                      str(z).rjust(12, ' ') + vel[0].rjust(8, ' ') + vel[1].rjust(8, ' ') + vel[2].rjust(9, ' '))
    atom_num += 1  
    return atom_num

def dummy_real_swap(mol_new, resnum, resname, df, vel, atom_num, orig_coords):
    """
    Add an atom to the file which is switching between dummy and real state or vice versa

    Parameters
    ----------
    mol_new : file
        The temporary file for the new GRO
    resnum : int
        The residue number of the atom being added
    resname : str
        The name of the residue which the atom being added belongs to
    df : pandas.DataFrame
        The master dataframe which contains the coordinates for the atom being added
    vel : list of float
        The velocity to assign to the atom being added
    atom_num : int
        The atom number to assign to the atom being added
    orig_coords : list of float
        The coordinates for the atom being added
    
    Returns
    -------
    line_num : int
        Since the atom may be added in a different order than it was in the previous file save the line number so that we skip it when we come to it
    """
    #Get the name for the atom we are writting
    name_orig = df['Name'].to_list()[0]
    
    #Determine the new atom name based on whether the new atom should be real or dummy
    final_state = df['Final Type'].to_list()[0]
    if final_state == 'dummy':
        element = name_orig.strip('0123456789')
        atom_name_num = name_orig.strip('DCVH')
        #Add D or C
        if element == 'C':
            name_new = f'D{element}{atom_name_num}'
        else:
            name_new = f'{element}V{atom_name_num}'
    else:
        if 'DC' in name_orig:
            name_new = name_orig.replace('DC', 'C')
        else:
            name_new = name_orig.replace('HV', 'H')

    #These may be added out of order so lets make sure we have the right coordinates
    line_num = df['File line'].to_list()[0]
    c = line_num - 2
    orig_coor = orig_coords[c]
    x, y, z = ["{:.7f}".format(orig_coor[0]), "{:.7f}".format(orig_coor[1]), "{:.7f}".format(orig_coor[2])]

    #Write line
    mol_new.write(f'{resnum}{resname}'.rjust(8, ' ') + name_new.rjust(7, ' ') + str(atom_num).rjust(5, ' ') + str(x).rjust(12, ' ')  + str(y).rjust(12, ' ') + 
                      str(z).rjust(12, ' ') + vel[0].rjust(8, ' ') + vel[1].rjust(8, ' ') + vel[2].rjust(9, ' '))
    return line_num

def sep_merge(line):
    """
    Seperate two GRO file columns which no longer have a space sperating them

    Parameters
    ----------
    line : list of str
        The line contents seperated by spaces
    
    Returns
    -------
    temp_line : list of str
        The line contents with the merged columns 
    """
    temp_line = [line[0]]
    merged = [*line[1]]
    atom_name = "".join(merged[:-5])
    atom_num = "".join(merged[-5:])
    temp_line.append(atom_name)
    temp_line.append(atom_num)
    for n in line[2:]:
        temp_line.append(n)
    return temp_line

def rotate_point_around_axis(point, vertex, axis, angle):
    """
    Rotate a 3D point around an aritrary axis
    
    Parameters
    ----------
    point : np.array shape (1,3)
        The starting point prior to rotation
    start : np.array shape (1,3)
        The vertex of the axis of rotation
    axis : np.array shape (1,3)
        The axis of rotation
    angle : float
        The angle to rotate

    Returns
    -------
    rotated point : np.array shape (1,3)
        The new rotated point
    """
    #Transformation matrix to origin
    T = np.eye(4)
    T[0:3,3] = -vertex
    #Reverse the transformation to origin
    T_r = np.eye(4)
    T_r[0:3,3] = vertex
    #Normalize rotational axis
    axis = axis / np.linalg.norm(axis)
    a, b, c = axis
    d = np.sqrt(b**2+c**2)

    # Rotation about the x axis
    Rx = [[1, 0, 0, 0], [0, c/d, -b/d, 0], [0, b/d, c/d, 0], [0, 0, 0, 1]]
    Rx_r = [[1, 0, 0, 0], [0, c/d, b/d, 0], [0, -b/d, c/d, 0], [0, 0, 0, 1]]

    #Rotation about the y axis
    Ry = [[d, 0, -a, 0], [0, 1, 0, 0], [a, 0, d, 0], [0, 0, 0, 1]]
    Ry_r = [[d, 0, a, 0], [0, 1, 0, 0], [-a, 0, d, 0], [0, 0, 0, 1]]

    #Rotation about the z axis
    Rz = [[np.cos(angle), np.sin(angle), 0, 0], [-np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    #Initial position 
    x0, y0, z0 = point
    init_arry = [x0, y0, z0, 1]

    #Perform the transofmration
    final_arry = T_r@Rx_r@Ry_r@Rz@Ry@Rx@T@init_arry
    
    #Get the corrdinates for the rotated point
    rotated_point = final_arry[0:3]

    return rotated_point

def find_rotation_angle(initial_point, vertex, rotated_point, axis):
    """
    Determine the angle of rotation around an arbitrary axis in 3D space.
    
    Parameters
    ----------
    initial_point: A numpy array
        the initial point coordinates
    rotated_point: A numpy array
        the rotated point coordinates
    axis: A numpy array
        the axis of rotation
    
    Returns
    -------
    angle: int
        The angle of rotation in radians.
    """
    # Normalize axis and seperate coordinates
    u, v, w = axis / np.linalg.norm(axis)
    x0, y0, z0 = initial_point
    a, b, c = vertex
    
    #Determine the rotated point with the same bond length as the initial point
    rotated_point_axis = rotated_point - vertex
    norm_rot_point_vec = rotated_point_axis / np.linalg.norm(rotated_point_axis)
    final_bond_length = np.sqrt((x0-a)**2 + (y0-b)**2 + (z0-c)**2)
    rotated_point_standard = vertex + final_bond_length*norm_rot_point_vec
    x, y, z = rotated_point_standard  #Change to point with same bond length along this vector

    Q = np.sqrt(v**2+w**2)
    B = ((w/Q)*(y0-b)) - ((v/Q)*(z0-c))
    A = Q*(x0-a) - u*(((v/Q)*(y0-b)) + ((w/Q)*(z0-c)))
    C = u*(x0-a) + Q*(((v/Q)*(y0-b)) + ((w/Q)*(z0-c)))
    L = np.arctan2(B,A)

    # Determine the rotational angle to line-up point
    cos_arg = (x-a-u*C)/(Q*np.sqrt(A**2+B**2))
    if cos_arg > 1 or cos_arg < -1:
        if abs(cos_arg - 1) < .0001:
            cos_arg = 1
        elif abs(cos_arg + 1) < .0001:
            cos_arg = -1
        else:
            raise Exception(f'Invalid arccos argument: {(x-a-u*C)/(Q*np.sqrt(A**2+B**2))}')
    arc_cos = np.arccos(cos_arg)
    angle = L + arc_cos

    #Test point
    new_pt = rotate_point_around_axis(initial_point, vertex, axis / np.linalg.norm(axis), angle)
    if abs(new_pt[1] - y) < 0.0001:
        return angle
    else:
        arc_cos = 2*np.pi - arc_cos
        angle = L + arc_cos
        if angle > 2*np.pi:
            angle = angle - 2*np.pi

    return angle

def compute_angle(coords):
    """
    Compute the angle between two vectors

    Parameters
    ----------
    coords : np.array
        Four points which define two vectors to compute the angle between

    Return
    ------
    angle : int
        Angle in radians between the two points
    """
    vec1 = coords[0] - coords[1]
    vec2 = coords[2] - coords[1]

    angle = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    return angle

def add_or_swap(df_select, file_new, resnum, resname, vel, atom_num, orig_coor, skip_line, R_o_D_num, pick):
    """
    Determine if the atom needs added or swapped between real and dummy state and then add the atom to the new file

    Parameters
    ----------
    df_select : pd.DataFrame

    file_new : File
        The temporary file for the new GRO
    resnum : int
        The residue number of the atom being added
    resname : str
        The name of the residue which the atom being added belongs to
    vel : list of float
        The velocity to assign to the atom being added
    atom_num : int
        The atom number to assign to the atom being added
    orig_coords : list of float
        The coordinates for the atom being added
    skip_line : list of str
        List of line numbers that should be skipped when we come across them while reading the file
    R_o_D_num : list of str
        list of atoms that need added
    pick : int
        Index in the list to add now
    Return
    ------
    skip_line : list of int
        Updated list of line numbers that should be skipped when we come across them while reading the file
    R_o_D_num_new : 
        Updated list of atoms that need added
    """
    c = df_select.index.values.tolist()
    if df_select.loc[c[0], 'Direction'] == 'miss': #Add atom if missing
        add_atom(file_new, resnum, resname, df_select, vel, atom_num)
    else: #Swap from dummy to real
        line = dummy_real_swap(file_new, resnum, resname, df_select, vel, atom_num, orig_coor) #Add the dummy atom from A as a real atom in B and save the line so it can be skipped later
        skip_line.append(line)
    R_o_D_num_new = np.delete(R_o_D_num, pick) #Atom no longer needs added
    return skip_line, R_o_D_num_new

def write_new_file(df_atom_swap, swap, r_swap, line_start, orig_file, new_file, old_res_name, new_res_name, orig_coords, miss):
    """
    Write a new GRO file

    Parameters
    ----------
    df_atom_swap : pd.DataFrame
        Master dataframe containing info on the atoms which will change before and after the swap
    swap : 
        Swapping direction
    r_swap : 
        Reverse of the swapping direction
    line_start : int
        The line number where we start reading the file
    orig_file : List of str
        Pre-swap file to read from
    new_file : File
        Temporary file to write new coordinates
    old_res_name : str
        Residue name from before the swap
    new_res_name : str
        Residue name for after the swap
    orig_coords : np.array
        Coordinates for all atoms in the system before the swap
    miss : list of str
        Residues which are needed after the swap which are not present before the swap

    Return
    ------
        None
    """
    #Add vero velocity to all atoms
    vel = ['0.000', '0.000', '0.000\n']

    atom_num_A, atom_num_B = 0, 0
    skip_line = []
    df_interest = df_atom_swap[((df_atom_swap['Swap'] == swap) & ((df_atom_swap['Direction'] == 'R2D') | (df_atom_swap['Direction'] == 'D2R'))) | ((df_atom_swap['Swap'] == r_swap) & (df_atom_swap['Direction'] == 'miss'))]
    R_C_num = df_interest[(df_interest['Element'] == 'C') & ((df_interest['Direction'] == 'D2R') | ((df_interest['Direction'] == 'miss') & (df_interest['Final Type'] == 'real')))]['Atom Name Number'].to_numpy(dtype=int)
    D_C_num = df_interest[(df_interest['Element'] == 'C') & ((df_interest['Direction'] == 'R2D') | ((df_interest['Direction'] == 'miss') & (df_interest['Final Type'] == 'dummy')))]['Atom Name Number'].to_numpy(dtype=int)
    R_H_num = df_interest[(df_interest['Element'] == 'H') & ((df_interest['Direction'] == 'D2R') | ((df_interest['Direction'] == 'miss') & (df_interest['Final Type'] == 'real')))]['Atom Name Number'].to_numpy(dtype=int)
    D_H_num = df_interest[(df_interest['Element'] == 'H') & ((df_interest['Direction'] == 'R2D') | ((df_interest['Direction'] == 'miss') & (df_interest['Final Type'] == 'dummy')))]['Atom Name Number'].to_numpy(dtype=int)
    R_C_num.sort()
    D_C_num.sort()
    R_H_num.sort()
    D_H_num.sort()
    for i in range(line_start, len(orig_file)-1):
        #Some atoms are added out of order from file A and thus must be skipped when they come up
        if i in skip_line:
            atom_num_A += 1
            continue
        #Iterate atom number to keep track of file progress
        atom_num_B += 1
        
        #Process input lines
        line, prev_line = process_line(orig_file, i)

        #Seperate resname from resnumber
        res_i = [*line[0]]
        prev_res_i = [*prev_line[0]]

        resname = "".join(res_i[-3:])
        resnum = "".join(res_i[0:int(len(res_i)-3)])
        prev_resname = "".join(prev_res_i[-3:])
        
        #Determine how to write line based on contents
        if resname != old_res_name and prev_resname != old_res_name: #Change only atom number and velocities if atoms not in residue of interest
            write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A])
        elif resname == old_res_name: #Atom manipulation required for acyl chain
            #determine the number for the atom being written in the current line
            if len(line[1]) > 1:
                current_num = int(line[1].strip('CVHDSON'))
            else:
                current_num = np.NaN
            current_element = line[1].strip('0123456789')
            if line[1] in miss: #Do not write coordinates if atoms are not present in B
                atom_num_B -= 1
                atom_num_A += 1
                continue
            elif (current_element == 'C' and current_num in D_C_num) or (current_element == 'H' and current_num in D_H_num): #Skip this line for now as we will add it in later
                atom_num_B -= 1
                atom_num_A += 1
                continue
            elif (current_element == 'C' and (current_num > R_C_num).any()): #We need to add an atom or move an atom to a new line position
                while (current_num > R_C_num).any():
                    #Either add atom or perform swap
                    pick = np.where(current_num > R_C_num)[0][0]
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(R_C_num[pick])) & (df_interest['Element'] == 'C')]
                    skip_line, R_C_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, R_C_num, pick)
                    atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            elif (len(R_C_num) != 0 and current_element == 'H'): #Add remaining real heavy atoms if all heavy atoms have been written
                while len(R_C_num) != 0:
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(R_C_num[0])) & (df_interest['Element'] == 'C')]
                    skip_line, R_C_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, R_C_num, 0)
                    atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            elif current_element == 'H' and (current_num > R_H_num).any(): #Add real Hs in proper order
                while (current_num > R_H_num).any():
                    #Either add atom or perform swap
                    pick = np.where(current_num > R_H_num)[0][0]
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(R_H_num[pick])) & (df_interest['Element'] == 'H')]
                    skip_line, R_H_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, R_H_num, pick)
                    atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            elif len(R_H_num) != 0 and (current_element == 'DC' or current_element == 'HV'): #Add remaining real Hs if all H atoms have been written
                while len(R_H_num) != 0:
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(R_H_num[0])) & (df_interest['Element'] == 'H')]
                    skip_line, R_H_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, R_H_num, 0)
                    atom_num_B += 1
                if current_element == 'DC':
                    while (current_num > D_C_num).any():
                        #Either add atom or perform swap
                        pick = np.where(current_num > D_C_num)[0][0]
                        df_select = df_interest[(df_interest['Atom Name Number'] == str(D_C_num[pick])) & (df_interest['Element'] == 'C')]
                        skip_line, D_C_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_C_num, pick)
                        atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            elif current_element == 'DC' and (current_num > D_C_num).any(): #Add dummy Cs in correct order
                while (current_num > D_C_num).any():
                    #Either add atom or perform swap
                    pick = np.where(current_num > D_C_num)[0][0]
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(D_C_num[pick])) & (df_interest['Element'] == 'C')]
                    skip_line, D_C_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_C_num, pick)
                    atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            elif len(D_C_num) != 0 and current_element == 'HV': #Add remaining dummy Cs after real Hs
                while len(D_C_num) != 0:
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(D_C_num[0])) & (df_interest['Element'] == 'C')]
                    skip_line, D_C_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_C_num, 0)
                    atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            elif current_element == 'HV' and (current_num > D_H_num).any(): #Add dummy Hs in proper order
                while (current_num > D_H_num).any():
                    #Either add atom or perform swap
                    pick = np.where(current_num > D_H_num)[0][0]
                    df_select = df_interest[(df_interest['Atom Name Number'] == str(D_H_num[0])) & (df_interest['Element'] == 'H')]
                    skip_line, D_H_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_H_num, 0)
                    atom_num_B += 1
                if i not in skip_line:
                    write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name) #Add this current line too
            else:
                write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A], resnum, new_res_name)
        elif resname != old_res_name and prev_resname == old_res_name: #Add dummy atoms at the end of the residue
            while len(R_H_num) != 0:
                df_select = df_interest[(df_interest['Atom Name Number'] == str(R_H_num[0])) & (df_interest['Element'] == 'H')]
                skip_line, R_H_num = add_or_swap(df_select, new_file, int(resnum)-1, new_res_name, vel, atom_num_B, orig_coords, skip_line, R_H_num, 0)
                atom_num_B += 1
            while len(D_C_num) != 0:
                df_select = df_interest[(df_interest['Atom Name Number'] == str(D_C_num[0])) & (df_interest['Element'] == 'C')]
                skip_line, D_C_num = add_or_swap(df_select, new_file, int(resnum)-1, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_C_num, 0)
                atom_num_B += 1
            while len(D_H_num) != 0:
                df_select = df_interest[(df_interest['Atom Name Number'] == str(D_H_num[0])) & (df_interest['Element'] == 'H')]
                skip_line, D_H_num = add_or_swap(df_select, new_file, int(resnum)-1, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_H_num, 0)
                atom_num_B += 1
            line, prev_line = process_line(orig_file, i)
            write_line(new_file, orig_file[i], line, atom_num_B, vel, orig_coords[atom_num_A])
        else:
            print(f'Warning line {i+1} not written')
        atom_num_A += 1
    
    if len(R_H_num) != 0: #If we hit end of coordinates without adding Hs add them now
        while len(R_H_num) != 0:
            atom_num_B += 1
            df_select = df_interest[(df_interest['Atom Name Number'] == str(R_H_num[0])) & (df_interest['Element'] == 'H')]
            skip_line, R_H_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, R_H_num, 0)

    if len(D_C_num) != 0: #If we hit end of coordinates without adding Cs add them now
        while len(D_C_num) != 0:
            atom_num_B += 1
            df_select = df_interest[(df_interest['Atom Name Number'] == str(D_C_num[0])) & (df_interest['Element'] == 'C')]
            skip_line, D_C_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_C_num, 0)

    if len(D_H_num) != 0: #If we hit end of coordinates without adding Hs add them now
        while len(D_H_num) != 0:
            atom_num_B += 1
            df_select = df_interest[(df_interest['Atom Name Number'] == str(D_H_num[0])) & (df_interest['Element'] == 'H')]
            skip_line, D_H_num = add_or_swap(df_select, new_file, resnum, new_res_name, vel, atom_num_B, orig_coords, skip_line, D_H_num, 0)
    
    #Add Box dimensions to file
    new_file.write(orig_file[-1])
    new_file.close()

def swap_name(init_atom_names, new_resname, df_top):
    """
    Determine the corresponding atom name in new molecule

    Parameters
    ----------
    init_atom_names : list of str
        Atom name in the original molecule
    new_resname : str
        Resname for the new molecule
    df_top : pd.DataFrame
        Dataframe containing the connectivity between the atoms in each molecule

    Return
    ------
    new_atom_names : list of str
        Atom names in the new molecule
    """
    #Find all atom names in new moleucle
    new_names = set(df_top[df_top['Resname'] == new_resname]['Connect 1'].to_list() + df_top[df_top['Resname'] == new_resname]['Connect 1'].to_list() + df_top[df_top['Resname'] == new_resname]['Connect 2'].to_list() + df_top[df_top['Resname'] == new_resname]['Connect 2'].to_list()) 
    new_atom_names = []
    for atom in init_atom_names:
        if atom in new_names:
            new_atom_names.append(atom)
            continue
        atom_num = re.findall(r'[0-9]+', atom)[0]
        atom_identifier = re.findall(r'[a-zA-Z]+', atom)[0]
        if 'D' in atom_identifier:
            atom_identifier = atom_identifier.strip('D')
        if 'V' in atom_identifier:
            atom_identifier = atom_identifier.strip('V')
        if f'{atom_identifier}V{atom_num}' in new_names:
            new_atom_names.append(f'{atom_identifier}V{atom_num}')
        elif f'{atom_identifier}{atom_num}' in new_names:
            new_atom_names.append(f'{atom_identifier}{atom_num}')
        elif f'D{atom_identifier}{atom_num}' in new_names:
            new_atom_names.append(f'D{atom_identifier}{atom_num}')
        else:
            raise Exception(f'Compatible atom could not be found for {atom}')
    return new_atom_names

def get_names(input):
    """
    Determine the names of all atoms in the topology and which lambda state for which they are dummy atoms

    Parameters
    ----------
    input : List of str
        Read the lines of the input topology

    Return
    ------
    start_line : int
        The next line to start reading from the topology
    atom_name : list of str
        All atom names in the topology
    state : list of int
        The state that the atom is a dummy atom (lambda=0, lambda=1, or -1 if nevver dummy)
    """
    atom_section = False
    atom_name, state = [], []
    for l, line in enumerate(input):
        if atom_section:
            line_sep = line.split(' ')
            if line_sep[0] == ';':
                continue
            while '' in line_sep:
                line_sep.remove('')
            if line_sep[0] == '\n':
                start_line = l+2
                break
            atom_name.append(line_sep[4])
            if len(line_sep) < 10:
                state.append(-1)
            elif float(line_sep[6]) == 0:
                state.append(0)
            elif float(line_sep[9]) == 0:
                state.append(1)
            else:
                state.append(-1)
        if line == '[ atoms ]\n':
            atom_section = True
    return start_line, atom_name, state

def deter_connection(main_only, other_only, main_name, other_name, df_top, main_state):
    """
    Determine the connectivity of the missing atoms in the topology

    Parameters
    ----------
    main_only : list of str
        All atoms which can be found only in the molecule of interest
    other_only : list of str
        All atoms which can be found only in the other molecule
    main_name : str
        resname for the molecule of interest
    other_name : str
        resname for the other molecule
    df_top : pd.DataFrame
        Connectivity of the atoms in each molecule
    main_state : list of int
        Which lambda state are each atom in the molecule of interest in the dummy state
    Return
    ------
    df : pd.DataFrame
        Dataframes contains the missing atoms, the real anchor atom that connects them, and the atom to be used to determine the angle to place the missing atoms
    """
    miss, D2R, R2D = [],[],[]
    align_atom, angle_atom = [], []
    for atom in main_only:
        element = atom.strip('0123456789')
        real_element = element.strip('DV')
        num = atom.strip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
        if f'D{atom}' in other_only or f'{element}V{num}' in other_only:
            D2R.append(atom)
        elif f'{real_element}{num}' in other_only:
            R2D.append(atom)
        else:
            miss.append(atom)
    df_select = df_top[df_top['Resname']==main_name]
    #Seperate into each seperate functional group to be added
    anchor_atoms  = []
    
    for m_atom in miss:
        #Find what atoms the missing atom connects to
        connected_atoms = []
        for a in df_select[df_select['Connect 1'] == m_atom]['Connect 2'].values:
            connected_atoms.append(a)
        for a in df_select[df_select['Connect 2'] == m_atom]['Connect 1'].values:
            connected_atoms.append(a)
        
        #If the atom connects to non-missing atoms than keep these as anchors
        for a in connected_atoms:
            if a not in miss:
                anchor_atoms.append(a)

    #Seperate missing atoms connected to each anchor
    miss_sep, align_atoms, angle_atoms = [],[],[]
    for anchor in anchor_atoms:
        miss_anchor = []

        #Which missing atoms are connected to the anchor
        search = True
        included_atoms = [anchor]
        while search == True:
            found_atoms = list(df_select[(df_select['Connect 1'].isin(included_atoms)) & (df_select['Connect 2'].isin(miss))]['Connect 2'].values) + list(df_select[(df_select['Connect 2'].isin(included_atoms)) & (df_select['Connect 1'].isin(miss))]['Connect 1'].values)
            if len(found_atoms) == 0:
                search = False
            else:
                for atom in found_atoms:
                    miss_anchor.append(atom)
                    included_atoms.append(atom)
                    miss.remove(atom)
        included_atoms.remove(anchor)
        miss_sep.append(included_atoms)
        
        #Find atoms connected to the anchor which are real in main state, but dummy when the atoms we are building are real
        align_atom = list(df_select[(df_select['Connect 1'] == anchor) & (df_select['State 2'] != main_state) & (df_select['State 2'] != -1)]['Connect 2'].values) + list(df_select[(df_select['Connect 2'] == anchor) & (df_select['State 1'] != main_state) & (df_select['State 1'] != -1)]['Connect 1'].values)
        align_atoms.append(align_atom[0])

        #Find the atom to use for matching the angle to ensure that the dummy atom is added in the correct orientation
        ignore_atoms = anchor_atoms + included_atoms + align_atom
        angle_atom = list(df_select[(df_select['Connect 1'] == anchor) & (~df_select['Connect 2'].isin(ignore_atoms))]['Connect 2'].values) + list(df_select[(df_select['Connect 2'] == anchor) & (~df_select['Connect 1'].isin(ignore_atoms))]['Connect 1'].values)
        angle_atoms.append(angle_atom[-1])
    
    #Now let's figure out what these atoms are called in the other molecule
    anchor_atoms_B = swap_name(anchor_atoms, other_name, df_top)
    angle_atoms_B = swap_name(angle_atoms, other_name, df_top)
    align_atoms_B = swap_name(align_atoms, other_name, df_top)
    df = pd.DataFrame({'Swap A': main_name, 'Swap B': other_name, 'Anchor Atom Name A': anchor_atoms, 'Anchor Atom Name B': anchor_atoms_B, 'Alignment Atom A': align_atoms, 'Alignment Atom B': align_atoms_B, 'Angle Atom A': angle_atoms, 'Angle Atom B': angle_atoms_B})
    miss_sep_reformat = []
    for missing in miss_sep:
        if isinstance(missing, list):
            missing_reformat = missing[0]
            for atom in missing[1:]:
                missing_reformat += f' {atom}'
        else:
            missing_reformat = missing
        miss_sep_reformat.append(missing_reformat)
    df['Missing Atom Name'] = miss_sep_reformat

    return df

def read_top(file_name, resname):
    """
    Read the topology to find the file containing the molecule of interest

    Parameters
    ----------
    file_name : str
        Name for the topology file
    resname : str
        Name of the residue of interest we are searching for

    Return
    ------
    input_file : list of str
        Contents of the topology file which contains the residue of interest
    """
    input_file = open(file_name).readlines()
    itp_files = []
    atom_sect = True
    for line in input_file:
        if line == '[ atoms ]\n':
            atom_sect = True
        if atom_sect:
            if line == '\n':
                atom_sect = False
            line_sep = line.split(' ')
            while '' in line_sep:
                line_sep.remove('') 
            if len(line_sep) > 4 and line_sep[3] == resname:
                return input_file
        if '#include' in line:
            line_sep = line.split(' ')
            while '' in line_sep:
                line_sep.remove('') 
            itp_files.append(line_sep[-1].strip('\n""'))
    for file in itp_files:
        if os.path.exists(file):
            input_file = open(file).readlines()
            atom_sect = False
            for line in input_file:
                if line == '[ atoms ]\n':
                    atom_sect = True
                if atom_sect:
                    if line == '\n':
                        break
                    line_sep = line.split(' ')
                    while '' in line_sep:
                        line_sep.remove('') 
                    if len(line_sep) > 4 and line_sep[3] == resname:
                        return input_file
    raise Exception(f'Residue {resname} can not be found in {file_name}')
