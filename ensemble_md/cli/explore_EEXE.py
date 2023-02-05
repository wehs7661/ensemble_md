####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
import sys
import argparse
import numpy as np
import pandas as pd


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This code explores the parameter space of homogenous EEXE to help you figure \
                out all possible combinations of the number of replicas, the number of \
                states in each replica, and the number of overlapping states, and the total number states.')
    parser.add_argument('-N',
                        '--N',
                        required=True,
                        type=int,
                        help='The total number of states of the EEXE simulation.')
    parser.add_argument('-r',
                        '--r',
                        type=int,
                        help='The number of replicas that compose the EEXE simulation.')
    parser.add_argument('-n',
                        '--n',
                        type=int,
                        help='The number of states for each replica.')
    parser.add_argument('-s',
                        '--s',
                        type=int,
                        help='The state shift between adjacent replicas.')
    args_parse = parser.parse_args(args)

    return args_parse


def solv_EEXE_diophantine(N):
    """
    Solves the general nonlinear Diophantine equation associated with the homogeneous EEXE
    parameters.

    Parameters
    ----------
    N : int
        The total number of states of the homogeneous EEXE of interesst.

    Returns
    -------
    soln_all : pd.DataFrame
        A pandas DataFrame that lists all the solutions of (N, r, n, s).
    """
    soln_all = []   # [N, r, n, s]
    r_list = range(2, N)
    for r in r_list:
        t = np.arange(int((r - N + 1) / r), 1)
        n = N + (r - 1) * (t - 1)
        s = 1 - t
        soln_all.extend([{'N': N, 'r': r, 'n': n[i], 's': s[i]} for i in range(len(t))])  # [N, r, n, s]
    soln_all = pd.DataFrame.from_dict(soln_all)  # turn the dict to pandas df, which is more convenient

    return soln_all


def main():
    # For now, we only consider homogenous EEXE simulations
    args = initialize(sys.argv[1:])
    print('Exploration of the EEXE parameter space')
    print('=======================================')
    print('[ EEXE parameters of interest ]')
    print('- N: The total number of states')
    print('- r: The number of replicas')
    print('- n: The number of states for each replica')
    print('- s: The state shift between adjacent replicas')

    # Enuerate all possible combinations of (N, r, n, s) even if any of r, n, s is given - it's easy/fast anyway.
    soln_all = solv_EEXE_diophantine(args.N)

    # Now filter the solutions
    if args.r is not None:
        soln_all = soln_all[soln_all['r'] == args.r]
    if args.n is not None:
        soln_all = soln_all[soln_all['n'] == args.n]
    if args.s is not None:
        soln_all = soln_all[soln_all['s'] == args.s]

    # Print out the solutions
    print('\n[ Solutions ]')
    for row_idx in range(len(soln_all)):
        soln = soln_all.iloc[row_idx]
        start_idx = [i * soln['s'] for i in range(soln['r'])]
        state_ranges = [list(np.arange(i, i + soln['n'])) for i in start_idx]
        print(f"- Solution {row_idx + 1}: (N, r, n, s) = ({args.N}, {soln['r']}, {soln['n']}, {soln['s']})")
        for i in range(soln['r']):
            print(f'  - Replica {i}: {state_ranges[i]}')
        print()
