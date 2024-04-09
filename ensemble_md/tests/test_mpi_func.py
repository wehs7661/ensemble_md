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
Unit tests for the functions that use MPI, including `_run_grompp`, `_run_mdrun` and `run_REXEE`.
"""
import os
import yaml
import glob
import shutil
import pytest
from mpi4py import MPI
from ensemble_md.replica_exchange_EE import ReplicaExchangeEE


@pytest.fixture
def params_dict():
    """
    Generates a dictionary containing the required REXEE parameters.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    REXEE_dict = {
        'gmx_executable': 'gmx',
        'gro': 'ensemble_md/tests/data/sys.gro',
        'top': 'ensemble_md/tests/data/sys.top',
        'mdp': 'ensemble_md/tests/data/expanded.mdp',
        'n_sim': 4,
        'n_iter': 10,
        's': 1,
        'working_dir': 'ensemble_md/tests/data',
    }
    yield REXEE_dict

    # Remove the file after the unit test is done.
    yml_file = f'params_{rank}.yaml'
    if os.path.isfile(yml_file):
        os.remove(yml_file)


def get_REXEE_instance(input_dict, rank, yml_file=None):
    """
    Saves a dictionary as a yaml file and use it to instantiate the ReplicaExchangeEE class.
    This version of the function creates a unique YAML file for each MPI process. This could
    avoid race conditions between MPI processes where one process reads the file before another
    finished writing it, or even worse, tries to read it while it's being written, leading to
    inconsistent or incomplete data being read.
    """
    if yml_file is None:
        yml_file = f'params_{rank}.yaml'
    with open(yml_file, 'w') as f:
        yaml.dump(input_dict, f)
    REXEE = ReplicaExchangeEE(yml_file)
    return REXEE


def get_gmx_cmd_from_output(output):
    """
    Given a GROMACS output file like a LOG file or `mdout.mdp`, extract the GROMACS command that was run.

    Parameters
    ----------
    output : str
        The path to the GROMACS output file.

    Returns
    -------
    cmd : str
        The GROMACS command that was run.
    flags : dict
        The flags and values that were used in the GROMACS command.
    """
    f = open(output, 'r')
    lines = f.readlines()
    f.close()

    n = -1
    cmd = None
    for l in lines:  # noqa: E741
        n += 1
        if 'Command line' in l:
            if lines[n + 1].startswith(';'):
                cmd = lines[n+1].split(';')[1].strip()
            else:
                cmd = lines[n+1].strip()
            break
    if cmd is None:
        raise ValueError(f'Could not find the GROMACS command in the file {output}.')

    flags = {}
    cmd_split = cmd.split(' ')
    for i in range(len(cmd_split)):
        if cmd_split[i].startswith('-'):
            flags[cmd_split[i]] = cmd_split[i+1]

    return cmd, flags


@pytest.mark.mpi
def test_run_grompp(params_dict):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params_dict['grompp_args'] = {'-maxwarn': '1'}

    # Case 1: The first iteration, i.e., n = 0
    n = 0
    swap_pattern = [1, 0, 2, 3]
    REXEE = get_REXEE_instance(params_dict, rank)

    # Below we set up files for testing
    if rank == 0:
        for i in range(params_dict['n_sim']):
            # Here we use the template mdp file since this is mainly for testing the function, not the GROMACS command.
            os.makedirs(f'{REXEE.working_dir}/sim_{i}/iteration_{n}')
            shutil.copy(REXEE.mdp, f'{REXEE.working_dir}/sim_{i}/iteration_{n}/expanded.mdp')

    REXEE._run_grompp(n, swap_pattern)

    comm.barrier()  # Wait for all MPI processes to finish

    # Check if the output files were generated, then clean up
    if rank == 0:
        for i in range(params_dict['n_sim']):
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/sys_EE.tpr') is True
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/mdout.mdp') is True

            # Here we check if the command executed was what we expected
            mdp = f'{REXEE.working_dir}/sim_{i}/iteration_0/expanded.mdp'
            gro = params_dict['gro']
            top = params_dict['top']
            tpr = f'{REXEE.working_dir}/sim_{i}/iteration_0/sys_EE.tpr'
            mdout = f'{REXEE.working_dir}/sim_{i}/iteration_0/mdout.mdp'
            cmd = f'{REXEE.gmx_executable} grompp -f {mdp} -c {gro} -p {top} -o {tpr} -po {mdout} -maxwarn 1'
            assert get_gmx_cmd_from_output(mdout)[0] == cmd

            shutil.rmtree(f'{REXEE.working_dir}/sim_{i}')

    # Case 2: Other iterations, i.e., n != 0
    n = 1  # For swap_pattern, we stick with [1, 0, 2, 3]
    REXEE = get_REXEE_instance(params_dict, rank)
    if rank == 0:
        for i in range(params_dict['n_sim']):
            os.makedirs(f'{REXEE.working_dir}/sim_{i}/iteration_{n}')
            os.makedirs(f'{REXEE.working_dir}/sim_{i}/iteration_{n-1}')
            shutil.copy(REXEE.mdp, f'{REXEE.working_dir}/sim_{i}/iteration_{n}/expanded.mdp')
            shutil.copy(REXEE.gro, f'{REXEE.working_dir}/sim_{i}/iteration_{n-1}/confout.gro')

    REXEE._run_grompp(n, swap_pattern)

    # Check if the output files were generated, then clean up
    if rank == 0:
        for i in range(params_dict['n_sim']):
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_1/sys_EE.tpr') is True
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_1/mdout.mdp') is True

            # Here we check if the command executed was what we expected
            mdp = f'{REXEE.working_dir}/sim_{i}/iteration_1/expanded.mdp'
            gro = f'{REXEE.working_dir}/sim_{swap_pattern[i]}/iteration_0/confout.gro'
            top = params_dict['top']
            tpr = f'{REXEE.working_dir}/sim_{i}/iteration_1/sys_EE.tpr'
            mdout = f'{REXEE.working_dir}/sim_{i}/iteration_1/mdout.mdp'
            cmd = f'{REXEE.gmx_executable} grompp -f {mdp} -c {gro} -p {top} -o {tpr} -po {mdout} -maxwarn 1'
            assert get_gmx_cmd_from_output(mdout)[0] == cmd

            shutil.rmtree(f'{REXEE.working_dir}/sim_{i}')


@pytest.mark.mpi
def test_REXEE(params_dict):
    # This should also tests _run_grompp and _run_mdrun
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    params_dict['grompp_args'] = {'-maxwarn': '1'}
    params_dict['runtime_args'] = {'-nsteps': '10'}  # This will overwrite nsteps in mdp
    n = 0
    swap_pattern = [1, 0, 2, 3]
    REXEE = get_REXEE_instance(params_dict, rank)

    # Below we set up files for testing
    if rank == 0:

        print(glob.glob(f'{REXEE.working_dir}/*'))
        for i in range(params_dict['n_sim']):
            # Here we use the template mdp file since this is mainly for testing the function, not the GROMACS command.
            os.makedirs(f'{REXEE.working_dir}/sim_{i}/iteration_{n}')
            shutil.copy(REXEE.mdp, f'{REXEE.working_dir}/sim_{i}/iteration_{n}/expanded.mdp')
        assert os.path.isfile(f'{REXEE.working_dir}/sim_0/iteration_0/expanded.mdp') is True

    REXEE.run_REXEE(n, swap_pattern)

    comm.barrier()  # Wait for all MPI processes to finish

    # Check if the output files were generated, then clean up
    if rank == 0:
        for i in range(params_dict['n_sim']):
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/sys_EE.tpr') is True
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/mdout.mdp') is True
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/confout.gro') is True  # check if mdrun succeeded  # noqa: E501

            log = f'{REXEE.working_dir}/sim_{i}/iteration_0/md.log'
            cmd = f'{REXEE.gmx_executable} mdrun -s sys_EE.tpr -nsteps 10'
            assert get_gmx_cmd_from_output(log)[0] == cmd

            shutil.rmtree(f'{REXEE.working_dir}/sim_{i}')
