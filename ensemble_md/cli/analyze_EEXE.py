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
warnings.simplefilter(action='ignore', category=UserWarning)

from ensemble_md.utils import utils  # noqa: E402
from ensemble_md.analysis import analyze_traj  # noqa: E402
from ensemble_md.analysis import analyze_matrix  # noqa: E402
from ensemble_md.analysis import msm_analysis  # noqa: E402
from ensemble_md.analysis import calc_free_energy  # noqa: E402
from ensemble_md.ensemble_EXE import EnsembleEXE  # noqa: E402


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This code analyzes an ensemble of expanded ensemble')
    parser.add_argument('-y',
                        '--yaml',
                        type=str,
                        default='params.yaml',
                        help='The input YAML file used to run the EEXE simulation. (Default: params.yaml)')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='analyze_EEXE_log.txt',
                        help='The output log file that contains the analysis results of EEXE. (Default: analyze_EEXE_log.txt)')
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

    # Section 1. Analysis based on transitions betwee replicas
    print('[ Section 1. Analysis based on transitions between replicas ]')
    
    # 1-0. Read in replica-space trajectories
    print('1-0. Reading in rep_trajs.npy ...')
    rep_trajs = np.load('rep_trajs.npy')  # Shape: (n_sim, n_iter)

    # 1-1. Plot the replica-sapce trajectory
    print('1-1. Plotting transitions between replicas ...')
    dt_swap = EEXE.nst_sim * EEXE.dt    # dt for swapping replicas
    analyze_traj.plot_rep_trajs(rep_trajs, 'rep_trajs.png', dt_swap)

    # 1-2. Plot the replica transition matrix
    # Note that 1-2 and 1-3 assume lag time = 1. Will probably change this.
    print('1-2. Plotting the replica transition matrix (considering all configurations) ...')
    counts = [analyze_traj.traj2transmtx(rep_trajs[i], EEXE.n_sim, normalize=False) for i in range(len(rep_trajs))]
    reps_mtx = np.sum(counts, axis=0)  # First sum up the counts. This should be symmetric if n_ex=1. Otherwise it might not be. # noqa: E501
    reps_mtx /= np.sum(reps_mtx, axis=1)[:, None]   # and then normalize each row
    analyze_matrix.plot_matrix(reps_mtx, 'rep_transmtx_allconfigs.png')

    # 1-3. Calculate the spectral gap for the replica transition amtrix
    spectral_gap = analyze_matrix.calc_spectral_gap(reps_mtx)
    print(f'1-3. The spectral gap of the replica transition matrix: {spectral_gap:.3f}')

    # Section 2. Analysis based on transitions between states
    print('\n[ Section 2. Analysis based on transitions between replicas ]')

    # 2-0. Stitch the trajectories for each configuration
    """
    print('2-0. Stitching trajectories for each configuration from dhdl files ...')
    dhdl_files = [natsort.natsorted(glob.glob(f'sim_{i}/iteration_*/*dhdl*xvg')) for i in range(EEXE.n_sim)]
    shifts = np.arange(EEXE.n_sim) * EEXE.s
    state_trajs = analyze_traj.stitch_trajs(dhdl_files, rep_trajs, shifts)  # length: the number of replicas
    print('     Saving state_trajs.npy ...')
    np.save('state_trajs.npy', state_trajs)   # save the stithced trajectories
    """
    state_trajs = np.load('state_trajs.npy')
    # 2-1. Plot the state-space trajectory
    state_trajs = np.load('state_trajs.npy')
    print('\n2-1. Plotting transitions between different alchemical states ...')
    dt_traj = EEXE.dt * EEXE.template['nstdhdl']  # in ps
    analyze_traj.plot_state_trajs(state_trajs, EEXE.state_ranges, 'state_trajs.png', dt_traj)

    # 2-2. Plot the implied timescale as a function of lag time
    print('\n2-2. Plotting the implied timescale as a function of lag time for all configurations ...')
    lags = np.arange(EEXE.lag_spacing, EEXE.lag_max + EEXE.lag_spacing, EEXE.lag_spacing)
    # lags could also be None and decided automatically. Could consider using that.
    ts_list = msm_analysis.plot_its(state_trajs, lags, fig_name='implied_timescales.png', dt=dt_traj, units='ps')

    # 2-3. Decide a lag time for building MSM
    print('\n2-3. Calculating recommended lag times for building MSMs ...')
    chosen_lags = msm_analysis.decide_lagtimes(ts_list)
    print('     \nSuggested lag times for each configuration:')
    for i in range(len(chosen_lags)):
        print(f'       - Configuration {i}: {chosen_lags[i] * dt_traj:.1f} ps')

    # 2-4. Build a Bayesian MSM and perform a CK test for each configuration to validate the models
    # Note that steps 2-4. to 2-9 all require MSMs, while state 2-9 doesn't.
    print('\n2-4. Building Bayesian MSMs for the state-space trajectory for each configuration ...')
    print('     Performing a Chapman-Kolmogorov test on each trajectory ...')
    models = [pyemma.msm.bayesian_markov_model(state_trajs[i], chosen_lags[i], dt_traj=f'{dt_traj} ps', show_progress=False) for i in range(EEXE.n_sim)]  # noqa: E501

    for i in range(len(models)):
        print(f'     Plotting the CK-test results for configuration {i} ...')
        mlags = 5  # this maps to 5 (mlags: multiples of lag times for testing the model)
        nsets = models[i].nstates  # number of metastable states.
        # Note that nstates is the number of unique states in the input trajectores counted with the effective mode
        # (see the documentation) Therefore, if a system barely sampled some of the states, those states will not be
        # counted as involved in the transition matrix (i.e. not in the active set). To check the active states,
        # use models[i].active_set. If the system sampled all states frequently, models[i].active_set should be equal
        # to np.unique(state_trajs[i]) and both lengths should be EEXE.n_tot.
        # I'm not sure why the attribute nstates_full is not always EEXE.n_tot but is less relevant here.
        cktest = models[i].cktest(nsets=nsets, mlags=mlags, show_progress=False)
        pyemma.plots.plot_cktest(cktest, dt=dt_traj, units='ps')
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(f'config_{i}_CKtest.png', dpi=600)

    # Additionally, check if the sampling is poor for each configuration
    for i in range(len(models)):
        if models[i].nstates != EEXE.n_tot:
            print(f'     Note: The sampling of configuration {i} was poor.')

    # 2-5. Plot the state transition matrices estimated with the specified lag times in MSMs
    print('\n2-5. Plotting the overall transition matrices for all configurations and their average ...')
    mtx_list = [m.transition_matrix for m in models]
    mtx_list_modified = []  # just for plotting (if all trajs sampled the fulle range frequently, this will be the same as mtx_list)  # noqa: E501
    for i in range(len(mtx_list)):
        # check if each mtx in mtx_list spans the full alchemical range. (If the system did not visit certain states,
        # the dimension will be less than EEXE.n_tot * EEXE.n_tot. In this case, we add rows and columns of 0.
        # Note that the modified matrix will not be a transition matrix, so this is only for plotting. For later
        # analysis such as spectral gap calculation, we will just use the unmodified matrices.
        if mtx_list[i].shape != (EEXE.n_tot, EEXE.n_tot):  # add rows and columns of 0
            sampled = models[i].active_set
            missing = list(set(range(EEXE.n_tot)) - set(sampled))  # states not visited

            # figure out which end we should stack rows/columns to
            n_1 = sum(missing > max(sampled))   # add rows/columns to the end of large state indices
            n_2 = sum(missing < min(sampled))   # add rows/columns to the end of small state indices

            # Below we first add a column and a row and so on
            mtx = mtx_list[i]
            for j in range(n_1):
                d = mtx.shape[0]  # size of the square matrix
                mtx = np.hstack((mtx, np.zeros([d, 1])))
                mtx = np.vstack((mtx, np.zeros([1, d + 1])))
            for j in range(n_2):
                d = mtx.shape[0]  # size of the square matrix
                mtx = np.hstack((np.zeros([d, 1]), mtx))
                mtx = np.vstack((np.zeros([1, d + 1]), mtx))
            mtx_list_modified.append(mtx)
        else:
            mtx_list_modified.append(mtx_list[i])

    avg_mtx = np.mean(mtx_list_modified, axis=0)
    print('     Saving transmtx.npy (plotted transition matrices)...')
    np.save('transmtx.npy', mtx_list_modified)

    for i in range(EEXE.n_sim):
        analyze_matrix.plot_matrix(mtx_list[i], f'config_{i}_state_transmtx.png')
    analyze_matrix.plot_matrix(avg_mtx, 'state_transmtx_avg.png')

    # Note that if a configuration never visited a certain replica, the rows in the alchemical range of that
    # replica will only have 0 entries. Below we check if this is the case.
    print('     Checking the sum of each row of each transition matrix is 1 ...')
    is_transmtx = [is_transition_matrix(i) for i in mtx_list]
    for i in range(len(is_transmtx)):
        if is_transmtx[i] is False:
            print(f'     The sums of the rows are not 1 (but {np.sum(mtx_list[i], axis=1)}) for the transition matrix of configuration {i}.')  # noqa: E501

    # 2-6. Calculate the spectral gap from the transition matrix of each configuration
    print('\n2-6. Calculating the spectral gap of the state transition matrices ...')
    spectral_gaps = [analyze_matrix.calc_spectral_gap(mtx) for mtx in mtx_list]
    for i in range(EEXE.n_sim):
        print(f'       - Configuration {i}: {spectral_gaps[i]:.3f}')

    # 2-7. Calculate the stationary distribution for each configuration
    print('\n2-7. Calculating the stationary distributions ...')
    pi_list = [m.pi for m in models]
    for i in range(EEXE.n_sim):
        print(f'       - Configuration {i}: {", ".join([f"{j:.3f}" for j in pi_list[i]])}')
    if len({len(i) for i in pi_list}) == 1:  # all lists in pi_list have the same length
        print(f'       - Average of the above: {", ".join([f"{i:.3f}" for i in np.mean(pi_list, axis=0)])}')

    # 2-8. Calculate the mean first-passage time (MFPT) from each MSM
    print('\n2-8. Calculating the mean first-passage times (MFPTs) ...')
    # note that it's not m.mfpt(min(m.active_set), max(m.active_set)) as the input to mfpt should be indices
    # though sometimes these two could be same.
    mfpt_list = [m.mfpt(0, m.nstates - 1) for m in models]
    for i in range(EEXE.n_sim):
        print(f'       - Configuration {i}: {mfpt_list[i]:.1f} ps')
    print(f'       - Average of the above: {np.mean(mfpt_list):.1f} ps')

    # 2-9. Calculate the state index correlation time for each configuration
    print('\n2-9. Plotting the state index correlation times for all configurations ...')
    msm_analysis.plot_acf(models, EEXE.n_tot, 'state_ACF.png')

    # 2-10. Calculate the end-to-end transit time for each configuration
    print('\n2-10. Plotting average end-to-end transit time')
    t_transit_list, units = analyze_traj.plot_transit_time(state_trajs, EEXE.n_tot, 'transit_time.png', dt_traj)
    for i in range(EEXE.n_sim):
        if t_transit_list[i] is None:
            print(f'       - Configuration {i}: Skipped')
        else:
            print(f'       - Configuration {i}: {np.mean(t_transit_list[i]):.1f} {units}')
    if None not in t_transit_list:
        print(f'       - Average of the above: {np.mean([np.mean(i) for i in t_transit_list]):.1f} {units}')

    # Section 3. Free energy calculations
    # TODO: Note that if no weight combination is used, g_vecs will be a list of None. This needs to be fixed.
    print('\n[ Section 3. Free energy calculations ]')
    print('3-1. Loading in g_vecs.npy')
    g_vecs = np.load('g_vecs.npy', allow_pickle=True)  # to ensure data can be read even if it's a list of None
    print('3-2. Calculating the free energy difference between two end states ...')
    if None not in g_vecs:
        dg_avg, dg_avg_err = calc_free_energy.average_dg(g_vecs, frac=EEXE.fe_frac, n_boots=EEXE.fe_boots)
        print(f'     The free energy difference averaged over the last {EEXE.frac * 100} percent of simulation: {dg_avg} +/- {dg_avg_err} kT.')  # noqa: E501
    else:
        print('No free energy calculations available as no weight combination was used. This will be allowed in future updates.')  # noqa: E501

    print(f'\nTime elpased: {utils.format_time(time.time() - t0)}')