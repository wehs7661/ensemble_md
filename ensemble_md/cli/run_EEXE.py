####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
import os
import sys
import glob
import time
import copy
import shutil
import argparse
import numpy as np
from mpi4py import MPI
from datetime import datetime

from ensemble_md.utils import utils
from ensemble_md.ensemble_EXE import EnsembleEXE
from ensemble_md.utils.exceptions import ParameterError


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This code runs an ensemble of expanded ensemble given necessary inputs.')
    parser.add_argument('-y',
                        '--yaml',
                        type=str,
                        default='params.yaml',
                        help='The input YAML file that contains EEXE parameters. (Default: params.yaml)')
    parser.add_argument('-c',
                        '--ckpt',
                        type=str,
                        default='rep_trajs.npy',
                        help='The NPY file containing the replica-space trajectories. This file is a \
                            necessary checkpoint file for extending the simulaiton. (Default: rep_trajs.npy)')
    parser.add_argument('-g',
                        '--g_vecs',
                        type=str,
                        default='g_vecs.npy',
                        help='The NPY file containing the timeseries of the whole-range alchemical weights. \
                            This file is a necessary input if ones wants to update the file when extending \
                            the simulation. (Default: g_vecs.npy)')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='run_EEXE_log.txt',
                        help='The output file for logging how replicas interact with each other. \
                            (Default: run_EEXE_log.txt)')
    parser.add_argument('-m',
                        '--maxwarn',
                        type=int,
                        default=0,
                        help='The maximum number of warnings in parameter specification to be ignored.')
    args_parse = parser.parse_args(args)

    return args_parse


@profile
def main():
    t1 = time.time()
    args = initialize(sys.argv[1:])
    sys.stdout = utils.Logger(logfile=args.output)
    sys.stderr = utils.Logger(logfile=args.output)

    # Step 1: Set up MPI rank and instantiate EnsembleEXE to set up EEXE parameters
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Note that this is a GLOBAL variable

    if rank == 0:
        print(f'Current time: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        print(f'Command line: {" ".join(sys.argv)}\n')

    EEXE = EnsembleEXE(args.yaml)

    if rank == 0:
        # Print out simulation parameters
        EEXE.print_params()

        # Print out warnings and fail if needed
        for i in EEXE.warnings:
            print(f'\n{i}\n')

        if len(EEXE.warnings) > args.maxwarn:
            raise ParameterError(
                f"The execution failed due to warning(s) about parameter spcificaiton. Check the warnings, or consider setting maxwarn in the input YAML file if you find them harmless.")  # noqa: E501, F541

    # Step 2: If there is no checkpoint file found/provided, perform the 1st iteration (index 0)
    if os.path.isfile(args.ckpt) is False:
        # 2-1. Set up input files for all simulations with 1 rank
        if rank == 0:
            for i in range(EEXE.n_sim):
                os.mkdir(f'sim_{i}')
                os.mkdir(f'sim_{i}/iteration_0')
                MDP = EEXE.initialize_MDP(i)
                MDP.write(f"sim_{i}/iteration_0/{EEXE.mdp.split('/')[-1]}", skipempty=True)

        # 2-2. Run the first ensemble of simulations
        md = EEXE.run_EEXE(0)

        # 2-3. Restructure the directory (move the files from mdrun_0_i0_* to sim_*/iteration_0)
        if rank == 0:
            work_dir = md.output.directory.result()
            for i in range(EEXE.n_sim):
                if EEXE.verbose is True:
                    print(f'  Moving files from {work_dir[i].split("/")[-1]}/ to sim_{i}/iteration_0/ ...')
                    print(f'  Removing the empty folder {work_dir[i].split("/")[-1]} ...')
                for f in glob.glob(f'{work_dir[i]}/*'):
                    shutil.move(f, f'sim_{i}/iteration_0/')
                os.rmdir(work_dir[i])
        start_idx = 1
    else:
        if rank == 0:
            # If there is a checkpoint file, we see the execution as an extension of an EEXE simulation
            ckpt_data = np.load(args.ckpt)
            start_idx = len(ckpt_data[0])
            print(f'\nGetting prepared to extend the EEXE simulation from iteration {start_idx} ...')

            print('Deleting corrupted data ...')
            corrupted = glob.glob('gmxapi.commandline.cli*')  # corrupted iteration
            corrupted.extend(glob.glob('mdrun*'))
            for i in corrupted:
                shutil.rmtree(i)
            if len(corrupted) == 0:
                corrupt_bool = False

            for i in range(EEXE.n_sim):
                n_finished = len(next(os.walk(f'sim_{i}'))[1])  # number of finished iterations (the last might be initialized but corrupted though)  # noqa: E501
                if n_finished == EEXE.n_iter and corrupt_bool is False:
                    print('Extension aborted: The expected number of iterations have been completed!')
                    sys.exit()
                else:
                    print('Deleting data generated after the checkpoint ...')
                    for j in range(start_idx, n_finished):
                        print(f'  Deleting the folder sim_{i}/iteration_{j}')
                        shutil.rmtree(f'sim_{i}/iteration_{j}')

            # Read g_vecs.npy and rep_trajs.npy so that new data can be appended, if any.
            EEXE.rep_trajs = [list(i) for i in ckpt_data]
            if os.path.isfile(args.g_vecs) is True:
                EEXE.g_vecs = [list(i) for i in np.load(args.g_vecs)]
        else:
            start_idx = None

        start_idx = comm.bcast(start_idx, root=0)  # so that all the ranks are aware of start_idx

    t_iteration = []
    for i in range(start_idx, EEXE.n_iter):
        if rank == 0:
            t_it1 = time.time()
            # Step 3: Swap the coordinates
            # 3-1. For all the replica simulations,
            #   (1) Find the last sampled state and the corresponding lambda values from the DHDL files.
            #   (2) Find the final Wang-Landau incrementors and weights from the LOG files.
            dhdl_files = [f'sim_{j}/iteration_{i - 1}/dhdl.xvg' for j in range(EEXE.n_sim)]
            log_files = [f'sim_{j}/iteration_{i - 1}/md.log' for j in range(EEXE.n_sim)]
            states_ = EEXE.extract_final_dhdl_info(dhdl_files)
            wl_delta, weights_, counts = EEXE.extract_final_log_info(log_files)
            print()

            # 3-2. Identify swappable pairs, propose swap(s), calculate P_acc, and accept/reject swap(s)
            # Note after `get_swapping_pattern`, `states_` and `weights_` won't be necessarily
            # since they are updated by `get_swapping_pattern`. (Even if the function does not explicitly
            # returns `states_` and `weights_`, `states_` and `weights_` can still be different after
            # the use of the function.) Therefore, here we create copyes for `states_` and `weights_`
            # before the use of `get_swapping_pattern`, so we can use them in `histogram_correction`,
            # `combine_weights` and `update_MDP`.
            states = copy.deepcopy(states_)
            weights = copy.deepcopy(weights_)
            swap_pattern = EEXE.get_swapping_pattern(dhdl_files, states_, weights_)

            # 3-3. Calculate the weights averaged since the last update of the Wang-Landau incrementor.
            # Note that the averaged weights are used for histogram correction/weight combination.
            # For the calculation of the acceptance ratio (inside get_swapping_pattern), final weights should be used.
            if EEXE.N_cutoff != -1 or EEXE.w_combine is True:
                # Only when histogram correction/weight combination is needed.
                weights_avg, weights_err = EEXE.get_averaged_weights(log_files)

            # 3-4. Perform histogram correction/weight combination
            # Note that we never use final weights but averaged weights here.
            # The product of this step should always be named as "weights" to be used in update_MDP
            if wl_delta != [None for i in range(EEXE.n_sim)]:  # weight-updating
                print(f'\nCurrent Wang-Landau incrementors: {wl_delta}')
            if EEXE.N_cutoff != -1 and EEXE.w_combine is True:
                # perform both
                weights_avg = EEXE.histogram_correction(weights_avg, counts)
                weights, g_vec = EEXE.combine_weights(weights_avg)
                EEXE.g_vecs.append(g_vec)
            elif EEXE.N_cutoff == -1 and EEXE.w_combine is True:
                # only perform weight combination
                print('\nNote: No histogram correction will be performed.')
                weights, g_vec = EEXE.combine_weights(weights_avg)
                EEXE.g_vecs.append(g_vec)
            elif EEXE.N_cutoff != -1 and EEXE.w_combine is False:
                # only perform histogram correction
                print('\nNote: No weight combination will be performed.')
                weights_avg = EEXE.histogram_correction(weights_avg, counts)
            else:
                print('\nNote: No histogram correction will be performed.')
                print('Note: No weight combination will be performed.')

            # 3-5. Modify the MDP files and swap out the GRO files (if needed)
            # Here we keep the lambda range set in mdp the same across different iterations in the same folder but swap out the gro file  # noqa: E501
            # Note we use states (copy of states_) instead of states_ in update_MDP.
            for j in list(range(EEXE.n_sim)):
                os.mkdir(f'sim_{j}/iteration_{i}')
                MDP = EEXE.update_MDP(f"sim_{j}/iteration_{i - 1}/{EEXE.mdp.split('/')[-1]}", j, i, states, wl_delta, weights)   # modify with a new template  # noqa: E501
                MDP.write(f"sim_{j}/iteration_{i}/{EEXE.mdp.split('/')[-1]}", skipempty=True)
                # In run_EEXE(i, swap_pattern), where the tpr files will be generated, we use the top file at the
                # level of the simulation (the file that will be shared by all simulations). For the gro file, we pass
                # swap_patter to the function to figure it out internally.

        if -1 not in EEXE.equil and 0 not in EEXE.equil:
            # This is the case where the weights are equilibrated in a weight-updating simulation.
            # As a remidner, EEXE.equil should be a list of 0 after extract_final_log_info in a
            # fixed-weight simulation, and a list of -1 for a weight-updating simulation with unequilibrated weights.
            print('\nSimulation terminated: The weights have been equilibrated for all replicas.')
            break

        # Step 4: Perform another iteration
        # 4-1. Run another ensemble of simulations
        md = EEXE.run_EEXE(i, swap_pattern)

        if rank == 0:
            # 4-2. Restructure the directory (move the files from mdrun_{i}_i0_* to sim_*/iteration_{i})
            work_dir = md.output.directory.result()
            for j in range(EEXE.n_sim):
                if EEXE.verbose is True:
                    print(f'  Moving files from {work_dir[j].split("/")[-1]}/ to sim_{j}/iteration_{i}/ ...')
                    print(f'  Removing the empty folder {work_dir[j].split("/")[-1]} ...')
                for f in glob.glob(f'{work_dir[j]}/*'):
                    shutil.move(f, f'sim_{j}/iteration_{i}/')
                os.rmdir(work_dir[j])
            t_it2 = time.time()
            t_iteration.append(t_it2 - t_it1)

            # 4-3. Save data
            if (i + 1) % EEXE.n_ckpt == 0:
                if len(EEXE.g_vecs) != 0:
                    # Save g_vec as a function of time if weight combination was used.
                    np.save('g_vecs.npy', EEXE.g_vecs)

                print('\n----- Saving .npy files to checkpoint the simulation ---')
                np.save('rep_trajs.npy', EEXE.rep_trajs)

    # Save the npy files at the end of the simulation anyway.
    if rank == 0:
        if len(EEXE.g_vecs) != 0:  # The length will be 0 only if there is no weight combination.
            np.save('g_vecs.npy', EEXE.g_vecs)
        np.save('rep_trajs.npy', EEXE.rep_trajs)
        np.save('t_iteration.npy', t_iteration)
        np.save('t_grompp.npy', EEXE.t_grompp)
        np.save('t_mdrun.npy', EEXE.t_mdrun)

    # Step 5: Write a summary for the simulation ensemble
    if rank == 0:
        print('\nSummary of the simulation ensemble')
        print('==================================')
        print('Simulation status:')
        for i in range(EEXE.n_sim):
            if EEXE.fixed_weights is True:
                print(f'- Rep {i}: The weights were fixed throughout the simulation.')
            elif EEXE.equil[i] == -1:
                print(f'  - Rep {i}: The weights have not been equilibrated.')
            else:
                idx = int(np.floor(EEXE.equil[i] / (EEXE.dt * EEXE.nst_sim)))
                if EEXE.equil[i] > 1000:
                    units = 'ns'
                    EEXE.equil[i] /= 1000
                else:
                    units = 'ps'
                print(f'  - Rep {i}: The weights have been equilibrated at {EEXE.equil[i]:.2f} {units} (iteration {idx}).')  # noqa: E501

        print(f'\n{EEXE.n_empty_swappable} out of {EEXE.n_iter}, or {EEXE.n_empty_swappable / EEXE.n_iter * 100:.1f}% iterations had an empty list of swappable pairs.')  # noqa: E501
        print(f'{EEXE.n_rejected} out of {EEXE.n_swap_attempts}, or {EEXE.n_rejected / EEXE.n_swap_attempts * 100:.1f}% of attempted exchanges were rejected.')  # noqa: E501

        print(f'\nTime elapsed: {utils.format_time(time.time() - t1)}')

    sys.exit()
