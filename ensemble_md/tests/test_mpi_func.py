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
import shutil
import pytest
from mpi4py import MPI
from ensemble_md.replica_exchange_EE import ReplicaExchangeEE

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

@pytest.fixture
def params_dict():
    """
    Generates a dictionary containing the required REXEE parameters.
    """
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
    if os.path.isfile('params.yaml') is True:
        os.remove('params.yaml')

    # Remove the file after the unit test is done.
    if os.path.isfile('params.yaml') is True:
        os.remove('params.yaml')


def get_REXEE_instance(input_dict, yml_file='params.yaml'):
    """
    Saves a dictionary as a yaml file and use it to instantiate the ReplicaExchangeEE class.
    """
    with open(yml_file, 'w') as f:
        yaml.dump(input_dict, f)
    REXEE = ReplicaExchangeEE(yml_file)
    return REXEE


@pytest.mark.mpi
def test_run_grompp(params_dict):
    params_dict['grompp_args'] = {'-maxwarn': '1'}

    # Case 1: The first iteration, i.e., n = 0
    n = 0
    swap_pattern = [1, 0, 2, 3]
    REXEE = get_REXEE_instance(params_dict)

    if rank == 0:
        for i in range(params_dict['n_sim']):
            os.makedirs(f'{REXEE.working_dir}/sim_{i}/iteration_{n}')
            shutil.copy(REXEE.mdp, f'{REXEE.working_dir}/sim_{i}/iteration_{n}/expanded.mdp')

    REXEE._run_grompp(n, swap_pattern)

    # Check if the output files are generated, then clean up
    if rank == 0:
        for i in range(params_dict['n_sim']):
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/sys_EE.tpr') is True
            assert os.path.isfile(f'{REXEE.working_dir}/sim_{i}/iteration_0/mdout.mdp') is True
            shutil.rmtree(f'{REXEE.working_dir}/sim_{i}')