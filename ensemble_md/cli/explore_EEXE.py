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
import random
import argparse
import numpy as np
import pandas as pd
from ensemble_md.ensemble_EXE import EnsembleEXE


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
    parser.add_argument('-c',
                        '--cnst',
                        default=False,
                        action='store_true',
                        help='Whether the apply the constraint such that the number of overlapping states \
                            does not exceed 50%% of the number of states in both overlapping replicas.')
    parser.add_argument('-e',
                        '--estimate',
                        default=False,
                        action='store_true',
                        help='Whether to provide estimates of the chance of not having any swappable \
                            pairs for each solution.')
    args_parse = parser.parse_args(args)

    return args_parse


def solv_EEXE_diophantine(N, constraint=False):
    """
    Solves the general nonlinear Diophantine equation associated with the homogeneous EEXE
    parameters. Specifically, given the total number of states :math:`N` and the number of replicas
    r, the states for each replica n and the state shift s can be expressed as:
    n = N + (r-1)(t-1), and s = 1 - t, with the range of t being either the following:
    - Without the additional constraint, (r-N+1)/r <= t <= 0
    - With the additional constraint, (r-N+1)/r <= t <= (r-N+1)/(r+1)

    Parameters
    ----------
    N : int
        The total number of states of the homogeneous EEXE of interesst.
    constraint : bool
        Whether to apply additional constraints such that n-s <= 1/2n.

    Returns
    -------
    soln_all : pd.DataFrame
        A pandas DataFrame that lists all the solutions of (N, r, n, s).
    """
    soln_all = []   # [N, r, n, s]
    r_list = range(2, N)
    for r in r_list:
        if constraint is False:
            # To include the upper bound 0, we set `stop=1` instead of 0.
            t = np.arange(int((r - N + 1) / r), 1).astype(int)
        else:
            if (r - N + 1) / (r + 1) == int((r - N + 1) / (r + 1)):
                # If the upper bound is a negative integer, we need to shift `stop` by 1.
                t = np.arange(int((r - N + 1) / r), int((r - N + 1) / (r + 1)) + 1).astype(int)
            else:
                # If the upper bound is a negative floating number, there is no need to shift.
                t = np.arange(int((r - N + 1) / r), int((r - N + 1) / (r + 1))).astype(int)
        n = N + (r - 1) * (t - 1)
        s = 1 - t
        soln_all.extend([{'N': N, 'r': r, 'n': n[i], 's': s[i]} for i in range(len(t))])  # [N, r, n, s]
    soln_all = pd.DataFrame.from_dict(soln_all)  # turn the dict to pandas df, which is more convenient

    return soln_all


def estimate_swapless_rate(state_ranges, N=1000000):
    """
    Estimates the chance of not having any swappable pair in an iteration.

    Parameters
    ----------
    state_ranges : list
        A list of lists of state indices. (Like the attribute :code:`state_ranges` in :code:`EnsemblEXE`.)
    N : n
        The number of Monte Carlo iterations for the estimation.

    Returns
    -------
    p : float
        The probability of not having any swappable pair in an iteration.
    """
    n = 0  # number of times of not having any swappable pairs
    for i in range(N):
        rands = [random.choice(state_ranges[i]) for i in range(len(state_ranges))]
        swappables = EnsembleEXE.identify_swappable_pairs(rands, state_ranges, False)
        if swappables == []:
            n += 1

    p = n / N * 100   # in percentage

    return p


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
    soln_all = solv_EEXE_diophantine(args.N, constraint=args.cnst)

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

        if args.estimate is True:
            p = estimate_swapless_rate(state_ranges)
            print(f"- Solution {row_idx + 1}: (N, r, n, s) = ({args.N}, {soln['r']}, {soln['n']}, {soln['s']}), {p:.1f}% chance to not have any swappable pair")  # noqa: E501
        else:
            print(f"- Solution {row_idx + 1}: (N, r, n, s) = ({args.N}, {soln['r']}, {soln['n']}, {soln['s']})")
        for i in range(soln['r']):
            print(f'  - Replica {i}: {state_ranges[i]}')
        print()
