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
import time
import shutil
import argparse
import numpy as np
from mpi4py import MPI

from ensemble_md.utils import utils
from ensemble_md.ensemble_EXE import EnsembleEXE


def initialize(args):
    parser = argparse.ArgumentParser(
        description='This code runs an ensemble of expanded ensemble given necessary inputs.')
    parser.add_argument('-y',
                        '--yaml',
                        type=str,
                        default='params.yaml',
                        help='The input YAML file that contains EEXE parameters.')
    args_parse = parser.parse_args(args)

    return args_parse


def main():
    t1 = time.time()
    args = initialize(sys.argv[1:])

    # Step 1: Set up MPI rank and instantiate EnsembleEXE to set up EEXE parameters
    rank = MPI.COMM_WORLD.Get_rank()  # Note that this is a GLOBAL variable
    EEXE = EnsembleEXE(args.yaml)
    sys.stdout = utils.Logger(logfile=EEXE.output)
    sys.stderr = utils.Logger(logfile=EEXE.output)
    EEXE.print_params()

    # Step 2: Perform the 1st iteration (index 0)
    # 2-1. Set up input files for all simulations with 1 rank
    if rank == 0:
        for i in range(EEXE.n_sim):
            os.mkdir(f'sim_{i}')
            os.mkdir(f'sim_{i}/iteration_0')
            MDP = EEXE.initialize_MDP(i)
            MDP.write(f"sim_{i}/iteration_0/{EEXE.mdp.split('/')[-1]}", skipempty=True)
            shutil.copy(f'{EEXE.gro}', f"sim_{i}/iteration_0/{EEXE.gro.split('/')[-1]}")
            shutil.copy(f'{EEXE.top}', f"sim_{i}/iteration_0/{EEXE.top.split('/')[-1]}")

    # 2-2. Run the first ensemble of simulations
    md = EEXE.run_EEXE(0)

    # 2-3. Restructure the directory (move the files from mdrun_0_i0_* to sim_*/iteration_0)
    if rank == 0:
        work_dir = md.output.directory.result()
        for i in range(EEXE.n_sim):
            if EEXE.verbose is True:
                print(f'  Moving files from {work_dir[i].split("/")[-1]}/ to sim_{i}/iteration_0/ ...')
                print(f'  Removing the empty folder {work_dir[i].split("/")[-1]} ...')
            os.system(f'mv {work_dir[i]}/* sim_{i}/iteration_0/.')
            os.rmdir(work_dir[i])

    # Step 3: Swap the coordinates
    g_vecs = []
    for i in range(1, EEXE.n_iter):
        if rank == 0:
            # 3-1. For all the replica simulations,
            #   (1) Find the last sampled state and the corresponding lambda values from the DHDL files.
            #   (2) Find the final Wang-Landau incrementors and weights from the LOG files.
            dhdl_files = [f'sim_{j}/iteration_{i - 1}/dhdl.xvg' for j in range(EEXE.n_sim)]
            log_files = [f'sim_{j}/iteration_{i - 1}/md.log' for j in range(EEXE.n_sim)]
            states, lambda_vecs = EEXE.extract_final_dhdl_info(dhdl_files)
            wl_delta, weights, counts = EEXE.extract_final_log_info(log_files)

            # 3-2. Identify swappable pairs, propose swap(s), calculate P_acc, and accept/reject swap(s)
            swap_list = EEXE.propose_swaps(states)
            swap_pattern = EEXE.get_swapping_pattern(swap_list, dhdl_files, states, lambda_vecs, weights)

            # 3-3. Perform histogram correction for the weights as needed
            weights = EEXE.histogram_correction(weights, counts)

            # 3-4. Combine the weights. Note that this is just for initializing the next iteration and is indepdent of swapping itself.  # noqa: E501
            weights, g_vec = EEXE.combine_weights(weights, method=EEXE.w_scheme)
            g_vecs.append(g_vec)

            # 3-5. Modify the MDP files and swap out the GRO files (if needed)
            # Here we keep the lambda range set in mdp the same across different iterations in the same folder but swap out the gro file  # noqa: E501
            for j in list(range(EEXE.n_sim)):
                os.mkdir(f'sim_{j}/iteration_{i}')
                MDP = EEXE.update_MDP(f"sim_{j}/iteration_{i - 1}/{EEXE.mdp.split('/')[-1]}", j, i, states, wl_delta, weights)   # modify with a new template  # noqa: E501
                MDP.write(f"sim_{j}/iteration_{i}/{EEXE.mdp.split('/')[-1]}", skipempty=True)
                shutil.copy(f'{EEXE.top}', f"sim_{j}/iteration_{i}/{EEXE.top.split('/')[-1]}")

                # Now we swap out the GRO files as needed
                shutil.copy(f'sim_{swap_pattern[j]}/iteration_{i-1}/confout.gro', f"sim_{j}/iteration_{i}/{EEXE.gro.split('/')[-1]}")  # noqa: E501

        # Step 4: Perform another iteration
        # 4-1. Run another ensemble of simulations
        md = EEXE.run_EEXE(i)

        # 4-2. Restructure the directory (move the files from mdrun_{i}_i0_* to sim_*/iteration_{i})
        if rank == 0:
            work_dir = md.output.directory.result()
            for j in range(EEXE.n_sim):
                if EEXE.verbose is True:
                    print(f'  Moving files from {work_dir[j].split("/")[-1]}/ to sim_{j}/iteration_{i}/ ...')
                    print(f'  Removing the empty folder {work_dir[j].split("/")[-1]} ...')
                os.system(f'mv {work_dir[j]}/* sim_{j}/iteration_{i}/.')
                os.rmdir(work_dir[j])

    np.save('g_vecs.npy', g_vecs)
    np.save('rep_trajs.npy', EEXE.rep_trajs)

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
                print(f'  - Rep {i}: The weights have been equilibrated at {EEXE.equil[i]} {units} (iteration {idx}).')

        print(f'\n{EEXE.n_rejected} out of {EEXE.n_swap_attempts}, or {EEXE.n_rejected / EEXE.n_swap_attempts * 100:.1f}% of attempted exchanges were rejected.')  # noqa: E501

        print(f'\nTime elapsed: {utils.format_time(time.time() - t1)}')
