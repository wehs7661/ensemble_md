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
import argparse
import warnings
import numpy as np

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

    EEXE = EnsembleEXE(args.yaml)
    EEXE.print_params()

    print('\nData analysis of the simulation ensemble')
    print('========================================')
    print('- Transitions between replicas')
    print('    Reading in rep_trajs.npy ...')
    rep_trajs = np.load('rep_trajs.npy')  # Shape: (n_sim, n_iter) (Note that n iterations involve n-1 times swapping.)

    print('    Plotting replica transition matrices ...')
    rep_mtxs = [analyze_trajs.traj2transmtx(rep_trajs[i], EEXE.n_sim) for i in range(len(rep_trajs))]
    for i in range(len(rep_mtxs)):
        analyze_matrix.plot_matrix(rep_mtxs[i], f'rep_transmtx_config_{i}.png')  # mostly not symmetric b.c. this is for each configuration # noqa: E501

    # Get the replica transition matrix considering all configurations
    counts = [analyze_trajs.traj2transmtx(rep_trajs[i], EEXE.n_sim, normalize=False) for i in range(len(rep_trajs))]
    reps_mtx = np.sum(counts, axis=0)  # First sum up the counts. This should be symmetric if n_ex=1. Otherwise it might not be. # noqa: E501
    reps_mtx /= np.sum(reps_mtx, axis=1)[:, None]   # and then normalize each row
    analyze_matrix.plot_matrix(reps_mtx, 'rep_transmtx_allconfigs.png')

    # Below we only analyze the averaged transition matrix for assessing mixing between replicas
    # Note that if for any of the replicas, the sums of some rows are 0 due to restricted sampling,
    # then the averaged matrix won't be a transition matrix.
    # rep_model = pyemma.msm.markov_model(avg_rep_mtx)

    print('- Transitions between alchemical states')
    print(f'\nTime elpased: {utils.format_time(time.time() - t0)}')
