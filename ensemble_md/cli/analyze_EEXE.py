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
import time
import glob
import pyemma
import natsort
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from deeptime.markov.tools.analysis import is_transition_matrix

from ensemble_md.utils import utils
from ensemble_md.analysis import analyze_trajs
from ensemble_md.analysis import analyze_matrix
from ensemble_md.ensemble_EXE import EnsembleEXE


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This code analyzes an ensemble of expanded ensemble')
    parser.add_argument('-y',
                        '--yaml',
                        type=str,
                        default='params.yaml',
                        help='The input YAML file used to run the EEXE simulation.')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='analyze_EEXE_log.txt',
                        help='The output log file that contains the analysis results of EEXE.')
    args_parse = parser.parse_args(args)

    return args_parse


def main():
    t0 = time.time()
    args = initialize(sys.argv[1:])
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    sys.stdout = utils.Logger(logfile=args.output)
    sys.stderr = utils.Logger(logfile=args.output)

    rc('font', **{
        'family': 'sans-serif',
        'sans-serif': ['DejaVu Sans'],
    })
    # Set the font used for MathJax - more on this later
    rc('mathtext', **{'default': 'regular'})
    plt.rc('font', family='serif')

    EEXE = EnsembleEXE(args.yaml)
    EEXE.print_params()

    print('\nData analysis of the simulation ensemble')
    print('========================================')
    print('[ Transitions between replicas ]')
    print('Reading in rep_trajs.npy ...')

    # 1. Plot the transitions between replicas
    print('Plotting transitions between replicas ...')
    rep_trajs = np.load('rep_trajs.npy')  # Shape: (n_sim, n_iter)
    dt = EEXE.nst_sim * EEXE.dt    # dt for swapping replicas
    analyze_trajs.plot_rep_trajs(rep_trajs, 'rep_trajs.png', dt)

    # 2. Plot the replica transition matrix
    print('Plotting the replica transition matrix (considering all configurations) ...')
    counts = [analyze_trajs.traj2transmtx(rep_trajs[i], EEXE.n_sim, normalize=False) for i in range(len(rep_trajs))]
    reps_mtx = np.sum(counts, axis=0)  # First sum up the counts. This should be symmetric if n_ex=1. Otherwise it might not be. # noqa: E501
    reps_mtx /= np.sum(reps_mtx, axis=1)[:, None]   # and then normalize each row
    analyze_matrix.plot_matrix(reps_mtx, 'rep_transmtx_allconfigs.png')

    # 3. Calculate the spectral gap for the replica transition amtrix
    spectral_gap = analyze_matrix.calc_spectral_gap(reps_mtx)
    print(f'The spectral gap of the replica transition matrix: {spectral_gap:.3f}')

    print('\n[ Transitions between alchemical states ]')
    print('Stitching trajectories for each configuration from dhdl files ...')
    dhdl_files, log_files = [], []
    for i in range(EEXE.n_sim):
        dhdl_files.append(natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/*dhdl*xvg')))
        log_files.append(natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/*log')))

    # 4. Plot the transitions between states
    print('Plotting transitions between different alchemical states ...')
    dt = EEXE.dt * EEXE.template['nstdhdl']
    shifts = np.arange(EEXE.n_sim) * EEXE.s
    state_trajs = analyze_trajs.stitch_trajs(dhdl_files, rep_trajs, shifts)
    analyze_trajs.plot_state_trajs(state_trajs, EEXE.state_ranges, 'state_trajs.png', dt)

    # 5. Plot the state transition matrices
    print('Calculating and plotting the overall transition matrix for each configuration ...')
    print('Calculating and plotting the overall transition matrix averaged all configurations ...')
    mtx_list = [analyze_trajs.traj2transmtx(state_trajs[i], EEXE.n_tot) for i in range(EEXE.n_sim)]
    avg_mtx = np.mean(mtx_list, axis=0)

    for i in range(len(mtx_list)):
        analyze_matrix.plot_matrix(mtx_list[i], f'config_{i}_state_transmtx.png')
    analyze_matrix.plot_matrix(avg_mtx, 'state_transmtx_avg.png')

    # Note that it is possible that the matrix in mtx_list is not a transition matrix, though it should be
    # very rare in a long simulation. For example, if replica 2 never visited replica 1, it won't visited
    # the first few states at all, i.e. p_ij = 0 for small i and j. Here we first check if there is such a case.
    print('Checking the sum of each row of each transition matrix is 1 ...')
    is_transmtx = [is_transition_matrix(i) for i in mtx_list]
    for i in range(len(is_transmtx)):
        if is_transmtx[i] is False:
            print(f'The sums of the rows are not 1 (but {np.sum(mtx_list[i], axis=1)}) for the transition matrix of configuration {i}.')  # noqa: E501

    print('\nAnalyzing state transition matrices ...')
    # 6. Calculate the spectral gap of the transition matrix for each configuration
    print('  - Spectral gaps')
    for i in range(len(mtx_list)):
        if is_transmtx[i] is False:
            print(f'    - Configuration {i}: Skipped.')
        else:
            gamma = analyze_matrix.calc_spectral_gap(mtx_list[i])
            print(f'    - Configuration {i}: {gamma:.3f}')
    if np.sum(is_transition_matrix) == len(is_transmtx):
        gamma_avg = analyze_matrix.calc_spectral_gap(avg_mtx)
        print(f'    - Avearge: {gamma_avg:.3f}')
    print()

    # 7. Calculate the stationary distribution for each configuration
    print('  - Stationary distributions')
    model_list = [pyemma.msm.markov_model(mtx_list[i]) if is_transmtx[i] is True else None for i in range(len(mtx_list))]  # noqa: E501
    for i in range(len(mtx_list)):
        if is_transmtx[i] is False:
            print(f'    - Configuration {i}: Skipped.')
        else:
            print(f'    - Configuration {i}: {", ".join([f"{val:.3f}" for val in model_list[i].pi])}')

    if np.sum(is_transmtx) == len(is_transmtx):
        model_avg = pyemma.msm.markov_model(avg_mtx)
        print(f'  - Average: {", ".join([f"{i:.3f}" for i in model_avg.pi])}')
    print()  # add a blank line

    # 8. Calculate the average end-to-end transit time of the state index for each configuration
    print('  - Average end-to-end transit time')
    dt = EEXE.dt * EEXE.template['nstdhdl']
    t_transit_list, units = analyze_trajs.plot_transit_time(state_trajs, EEXE.n_tot, 'transit_time.png', dt)
    for i in range(len(t_transit_list)):
        if t_transit_list[i] is None:
            print(f'    - Configuration {i}: Skipped.')
        else:
            print(f'    - Configuration {i}: {np.mean(t_transit_list[i]):.1f} {units}')

    if None not in t_transit_list:
        print(f'    - Average: {np.mean(np.mean(t_transit_list)):.1f} {units}')

    # 9. Calculate the mean first passage time from one end state to the other (ref: ref: shorturl.at/auAJX)

    # 10. Calcuulate the correlation time of the state index for each configuration

    # 11. Plot relaxation time/implied timescale as a function of lag time

    # 12. Chapman-Kolmogorov test

    # 13. Free energy calculations

    print(f'\nTime elpased: {utils.format_time(time.time() - t0)}')
