import sys
import time
import argparse
import warnings
import numpy as np

import ensemble_md.utils as utils
import ensemble_md.analysis as analysis
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
    rep_trajs = np.load('rep_trajs.npy')  # the shape should be (n_sim, n_iter + 1)

    print('    Plotting replica transition matrices ...')
    rep_mtxs = [analysis.traj2transmtx(rep_trajs[i], EEXE.n_sim) for i in range(len(rep_trajs))]
    for i in range(len(rep_mtxs)):
        analysis.plot_matrix(rep_mtxs[i], f'rep_transmtx_config_{i}.png')
    avg_rep_mtx = np.mean(rep_mtxs, axis=0)  # Should be symmetric! (nescessarily true for each rep transmtx though.)
    analysis.plot_matrix(avg_rep_mtx, 'rep_transmtx_avg.png')

    # Below we only analyze the averaged transition matrix for assessing mixing between replicas
    # Note that if for any of the replicas, the sums of some rows are 0 due to restricted sampling,
    # then the averaged matrix won't be a transition matrix.
    # rep_model = pyemma.msm.markov_model(avg_rep_mtx)

    print('- Transitions between alchemical states')
    print(f'\nTime elpased: {utils.format_time(time.time() - t0)}')
