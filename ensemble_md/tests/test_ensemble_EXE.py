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
Unit tests for the module ensemble_EXE.py.
"""
import os
import sys
import yaml
import pytest
import numpy as np
import ensemble_md
import gmxapi as gmx
from ensemble_md.utils import gmx_parser
from ensemble_md.ensemble_EXE import EnsembleEXE
from ensemble_md.utils.exceptions import ParameterError

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


@pytest.fixture
def params_dict():
    """
    Generates a dictionary containing the required EEXE parameters.
    """
    EEXE_dict = {
        'parallel': False,
        'gro': 'ensemble_md/tests/data/sys.gro',
        'top': 'ensemble_md/tests/data/sys.top',
        'mdp': 'ensemble_md/tests/data/expanded.mdp',
        'n_sim': 4,
        'n_iter': 10,
        's': 1,
    }
    yield EEXE_dict

    # Remove the file after the unit test is done.
    if os.path.isfile('params.yaml') is True:
        os.remove('params.yaml')


def get_EEXE_instance(input_dict, yml_file='params.yaml'):
    """
    Saves a dictionary as a yaml file and use it to instantiate the EnsembleEXE class.
    """
    with open(yml_file, 'w') as f:
        yaml.dump(input_dict, f)
    EEXE = EnsembleEXE(yml_file)
    return EEXE


def check_param_error(EEXE_dict, param, match, wrong_val='cool', right_val=None):
    """
    Test if ParameterError is raised if a parameter is not well-defined.
    """
    EEXE_dict[param] = wrong_val
    with pytest.raises(ParameterError, match=match):
        get_EEXE_instance(EEXE_dict)
    EEXE_dict[param] = right_val  # so that EEXE_dict can be read without failing in later assertions

    return EEXE_dict


class Test_EnsembleEXE:
    def test_init(self, params_dict):
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.yaml == 'params.yaml'

    def test_set_params_error(self, params_dict):
        # 1. Required parameters
        del params_dict['parallel']
        with pytest.raises(ParameterError, match="Required parameter 'parallel' not specified in params.yaml."):
            get_EEXE_instance(params_dict)
        params_dict['parallel'] = False  # so params_dict can be read without failing in the assertions below

        # 2. Available options
        check_param_error(params_dict, 'w_scheme', "The specified weight combining scheme is not available. Available options include None, 'mean', 'geo-mean'/'geo_mean' and 'g-diff/g_diff'.")  # noqa: E501
        check_param_error(params_dict, 'mc_scheme', "The specified MC scheme is not available. Available options include 'same-state', 'metropolis', and 'metropolis-eq'.")  # noqa: E501
        check_param_error(params_dict, 'df_method', "The specified free energy estimator is not available. Available options include 'TI', 'BAR', and 'MBAR'.")  # noqa: E501
        check_param_error(params_dict, 'err_method', "The specified method for error estimation is not available. Available options include 'propagate', and 'bootstrap'.")  # noqa: E501

        # 3. Integer parameters
        check_param_error(params_dict, 'nst_sim', "The parameter 'nst_sim' should be an integer.")
        check_param_error(params_dict, 'n_ex', "The parameter 'n_ex' should be an integer.")
        check_param_error(params_dict, 'seed', "The parameter 'seed' should be an integer.")
        check_param_error(params_dict, 'n_sim', "The parameter 'n_sim' should be an integer.", 4.1, 4)

        # 4. Positive parameters
        check_param_error(params_dict, 'nst_sim', "The parameter 'nst_sim' should be positive.", -5, 500)
        check_param_error(params_dict, 'n_iter', "The parameter 'n_iter' should be positive.", 0, 10)

        # 5. Non-negative parameters
        check_param_error(params_dict, 'n_ex', "The parameter 'n_ex' should be non-negative.", -1)
        check_param_error(params_dict, 'N_cutoff', "The parameter 'N_cutoff' should be non-negative unless no histogram correction is needed, i.e. N_cutoff = -1.", -5)  # noqa: E501

        # 6. String parameters
        check_param_error(params_dict, 'mdp', "The parameter 'mdp' should be a string.", 3, 'ensemble_md/tests/data/expanded.mdp')  # noqa: E501

        # 7. Boolean parameters
        check_param_error(params_dict, 'parallel', "The parameter 'parallel' should be a boolean variable.", 3, False)

        # 8. nstlog > nst_sim
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))  # A perfect mdp file
        mdp['nstlog'] = 200
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        params_dict['nst_sim'] = 100
        with pytest.raises(ParameterError, match='The parameter "nstlog" should be equal to or smaller than "nst_sim" specified in the YAML file so that the sampling information can be parsed.'):  # noqa: E501
            get_EEXE_instance(params_dict)
        params_dict['nst_sim'] = 500
        os.remove(os.path.join(input_path, "expanded_test.mdp"))

        # 9. n_sub < 1
        params_dict['s'] = 5
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded.mdp'
        # Note that the parentheses are special characters that need to be escaped in regular expressions
        with pytest.raises(ParameterError, match=r'There must be at least two states for each replica \(current value: -6\). The current specified configuration \(n_tot=9, n_sim=4, s=5\) does not work for EEXE.'):  # noqa: E501
            get_EEXE_instance(params_dict)
        params_dict['s'] = 1

    def test_set_params_warnings(self, params_dict):
        # 1. Non-recognizable parameter in the YAML file
        params_dict['cool'] = 10
        EEXE = get_EEXE_instance(params_dict)
        warning = 'Warning: Parameter "cool" specified in the input YAML file is not recognizable.'
        assert warning in EEXE.warnings

        # 2. Warnings related to the mdp file
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))  # A perfect mdp file
        mdp['lmc_seed'] = 1000
        mdp['gen_seed'] = 1000
        mdp['symmetrized_transition_matrix'] = 'yes'
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        EEXE = get_EEXE_instance(params_dict)
        warning_1 = 'Warning: We recommend setting lmc_seed as -1 so the random seed is different for each iteration.'
        warning_2 = 'Warning: We recommend setting gen_seed as -1 so the random seed is different for each iteration.'
        warning_3 = 'Warning: We recommend setting symmetrized-transition-matrix to no instead of yes.'
        assert warning_1 in EEXE.warnings
        assert warning_2 in EEXE.warnings
        assert warning_3 in EEXE.warnings

        os.remove(os.path.join(input_path, "expanded_test.mdp"))

    def test_set_params(self, params_dict):
        # 0. Get an EEXE instance to test
        EEXE = get_EEXE_instance(params_dict)

        # 1. Check the required EEXE parameters
        assert EEXE.gro == "ensemble_md/tests/data/sys.gro"
        assert EEXE.top == "ensemble_md/tests/data/sys.top"
        assert EEXE.mdp == "ensemble_md/tests/data/expanded.mdp"
        assert EEXE.parallel is False
        assert EEXE.n_sim == 4
        assert EEXE.n_iter == 10
        assert EEXE.s == 1

        # 2. Check the default values of the parameters not specified in params.yaml
        assert EEXE.mc_scheme == "metropolis"
        assert EEXE.w_scheme is None
        assert EEXE.N_cutoff == 1000
        assert EEXE.n_ex == 0
        assert EEXE.verbose is True
        assert EEXE.runtime_args is None
        assert EEXE.n_ckpt == 100
        assert EEXE.msm is False
        assert EEXE.free_energy is False
        assert EEXE.df_spacing == 1
        assert EEXE.df_method == 'MBAR'
        assert EEXE.err_method == 'propagate'
        assert EEXE.n_bootstrap == 50
        assert EEXE.seed is None

        # 3. Check the MDP parameters
        assert EEXE.nsteps == 500
        assert EEXE.dt == 0.002
        assert EEXE.temp == 298
        assert EEXE.nst_sim == 500
        assert EEXE.fixed_weights is False

        # 4. Checked the derived parameters
        k = 1.380649e-23
        NA = 6.0221408e23
        assert EEXE.kT == k * NA * 298 / 1000
        assert EEXE.lambda_types == ['coul_lambdas', 'vdw_lambdas']
        assert EEXE.n_tot == 9
        assert EEXE.n_sub == 6
        assert EEXE.state_ranges == [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8], ]
        assert EEXE.equil == [-1, -1, -1, -1]  # will be zeros right after the first iteration if the weights are fixed
        assert EEXE.n_rejected == 0
        assert EEXE.n_swap_attempts == 0
        assert EEXE.rep_trajs == [[0], [1], [2], [3]]
        assert EEXE.g_vecs == []

        params_dict['df_method'] = 'MBAR'
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.get_u_nk is True
        assert EEXE.get_dHdl is False

        params_dict['df_method'] = 'BAR'
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.get_u_nk is False
        assert EEXE.get_dHdl is True

        params_dict['runtime_args'] = {'-nt': 16, 'ntomp': 8}
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.runtime_args == {'-nt': 16, 'ntomp': 8}

        assert EEXE.reformatted_mdp is False

    def test_set_params_edge_cases(self, params_dict):
        # In the previous unit tests, we have tested the case where fixed_weights is False. Here we test the others.
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))

        # 1. wl_scale is not specified
        del mdp['wl_scale']
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.fixed_weights is True

        # 2. wl_scale is not empty
        mdp['wl_scale'] = ''  # It will become an empty array array([]) after read by gmx_parser.abs(x)
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.fixed_weights is True

    def test_reformat_MDP(self, params_dict):
        # Note that the function reformat_MDP is called in set_params
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))
        mdp['cool-stuff'] = 10
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        EEXE = get_EEXE_instance(params_dict)  # this calss reformat_MDP internally
        assert EEXE.reformatted_mdp is True
        assert EEXE.template['cool_stuff'] == 10
        assert os.path.isfile(os.path.join(input_path, "expanded_test_backup.mdp"))

        os.remove(os.path.join(input_path, "expanded_test.mdp"))
        os.remove(os.path.join(input_path, "expanded_test_backup.mdp"))

    def test_map_lambda2state(self, params_dict):
        # Note that the function map_lambda2state is called in set_params
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.lambda_dict == {
            (0, 0): 0,
            (0.25, 0): 1,
            (0.5, 0): 2,
            (0.75, 0): 3,
            (1, 0): 4,
            (1, 0.25): 5,
            (1, 0.5): 6,
            (1, 0.75): 7,
            (1, 1): 8}
        assert EEXE.lambda_ranges == [
            [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25)],
            [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5)],
            [(0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75)],
            [(0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0)], ]

        # Here we test another combinations: only 'fep_lambdas'
        mdp = gmx_parser.MDP(os.path.join(input_path, "expanded.mdp"))
        del mdp['coul_lambdas']
        del mdp['vdw_lambdas']
        mdp['fep_lambdas'] = '0.0 0.2 0.4 0.6 0.8 1.0'
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        params_dict['mdp'] = 'ensemble_md/tests/data/expanded_test.mdp'
        params_dict['n_sim'] = 3
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.lambda_types == ['fep_lambdas']
        assert EEXE.lambda_dict == {
            (0.0,): 0, (0.2,): 1, (0.4,): 2, (0.6,): 3, (0.8,): 4, (1.0,): 5}
        assert EEXE.lambda_ranges == [
            [(0.0, ), (0.2,), (0.4,), (0.6,)], [(0.2,), (0.4,), (0.6,), (0.8,)], [(0.4,), (0.6,), (0.8,), (1.0,)]]

        # Still another combination: fep_lambdas, coul_lambdas, vdw_lambdas, restraint_lambdas
        mdp['coul_lambdas'] = '0.0 0.5 0.8 1.0 1.0 1.0'
        mdp['vdw_lambdas'] = '0.0 0.0 0.0 0.0 0.5 1.0'
        mdp['restraint_lambdas'] = '0.0 0.2 0.4 0.6 0.8 1.0'
        mdp.write(os.path.join(input_path, "expanded_test.mdp"))
        EEXE = get_EEXE_instance(params_dict)
        assert EEXE.lambda_types == ['fep_lambdas', 'coul_lambdas', 'vdw_lambdas', 'restraint_lambdas']
        assert EEXE.lambda_dict == {
            (0.0, 0.0, 0.0, 0.0): 0,
            (0.2, 0.5, 0.0, 0.2): 1,
            (0.4, 0.8, 0.0, 0.4): 2,
            (0.6, 1.0, 0.0, 0.6): 3,
            (0.8, 1.0, 0.5, 0.8): 4,
            (1.0, 1.0, 1.0, 1.0): 5}

        assert EEXE.lambda_ranges == [
            [(0.0, 0.0, 0.0, 0.0), (0.2, 0.5, 0.0, 0.2), (0.4, 0.8, 0.0, 0.4), (0.6, 1.0, 0.0, 0.6)],
            [(0.2, 0.5, 0.0, 0.2), (0.4, 0.8, 0.0, 0.4), (0.6, 1.0, 0.0, 0.6), (0.8, 1.0, 0.5, 0.8)],
            [(0.4, 0.8, 0.0, 0.4), (0.6, 1.0, 0.0, 0.6), (0.8, 1.0, 0.5, 0.8), (1.0, 1.0, 1.0, 1.0)]]

        os.remove(os.path.join(input_path, "expanded_test.mdp"))

    def test_print_params(self, capfd, params_dict):
        # capfd is a fixture in pytest for testing STDOUT
        EEXE = get_EEXE_instance(params_dict)
        EEXE.print_params()
        out_1, err = capfd.readouterr()
        L = ""
        L += "Important parameters of EXEE\n============================\n"
        L += f"Python version: {sys.version}\ngmxapi version: {gmx.__version__}\nensemble_md version: {ensemble_md.__version__}\n"  # noqa: E501
        L += "Simulation inputs: ensemble_md/tests/data/sys.gro, ensemble_md/tests/data/sys.top, ensemble_md/tests/data/expanded.mdp\n"  # noqa: E501
        L += "Verbose log file: True\nWhether the replicas run in parallel: False\n"
        L += "MC scheme for swapping simulations: metropolis\nScheme for combining weights: None\n"
        L += "Histogram cutoff: 1000\nNumber of replicas: 4\nNumber of iterations: 10\n"
        L += "Number of exchanges in one attempt: 0\n"
        L += "Length of each replica: 1.0 ps\nFrequency for checkpointing: 100 iterations\n"
        L += "Total number of states: 9\n"
        L += "Additional runtime arguments: None\n"
        L += "Alchemical ranges of each replica in EEXE:\n  - Replica 0: States [0, 1, 2, 3, 4, 5]\n"
        L += "  - Replica 1: States [1, 2, 3, 4, 5, 6]\n  - Replica 2: States [2, 3, 4, 5, 6, 7]\n"
        L += "  - Replica 3: States [3, 4, 5, 6, 7, 8]\n"
        assert out_1 == L

        EEXE.print_params(params_analysis=True)
        out_2, err = capfd.readouterr()
        L += "\nWhether to build Markov state models and perform relevant analysis: False\n"
        L += "Whether to perform free energy calculations: False\n"
        L += "The step to used in subsampling the DHDL data in free energy calculations, if any: 1\n"
        L += "The chosen free energy estimator for free energy calculations, if any: MBAR\n"
        L += "The method for estimating the uncertainty of free energies in free energy calculations, if any: propagate\n"  # noqa: E501
        L += "The number of bootstrap iterations in the boostrapping method, if used: 50\n"
        L += "The random seed to use in bootstrapping, if used: None\n"
        assert out_2 == L

    def test_initialize_MDP(self, params_dict):
        EEXE = get_EEXE_instance(params_dict)
        MDP = EEXE.initialize_MDP(2)  # the third replica
        assert MDP["nsteps"] == 500
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP["vdw_lambdas"], [0.00, 0.00, 0.00, 0.25, 0.50, 0.75]
                )
            ]
        )
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP["coul_lambdas"], [0.50, 0.75, 1.00, 1.00, 1.00, 1.00]
                )
            ]
        )
        assert all(
            [a == b for a, b in zip(MDP["init_lambda_weights"], [0, 0, 0, 0, 0, 0])]
        )

    def test_update_MDP(self, params_dict):
        new_template = "ensemble_md/tests/data/expanded.mdp"
        iter_idx = 3
        states = [2, 5, 7, 4]
        wl_delta = [0.4, 0.32, 0.256, 0.32]
        weights = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [3.48, 2.78, 3.21, 4.56, 8.79, 0.48],
            [8.45, 0.52, 3.69, 2.43, 4.56, 6.73], ]

        EEXE = get_EEXE_instance(params_dict)
        EEXE.equil = [-1, 1, 0, -1]  # i.e. the 3rd replica will use fixed weights in the next iteration
        MDP_1 = EEXE.update_MDP(
            new_template, 2, iter_idx, states, wl_delta, weights)  # third replica
        MDP_2 = EEXE.update_MDP(
            new_template, 3, iter_idx, states, wl_delta, weights)  # fourth replica

        assert MDP_1["tinit"] == MDP_2["tinit"] == 3
        assert MDP_1["nsteps"] == MDP_2["nsteps"] == 500
        assert MDP_1["init_lambda_state"] == 5
        assert MDP_2["init_lambda_state"] == 1
        assert (MDP_1["init_wl_delta"] == MDP_1["wl_scale"] == MDP_1["wl_ratio"] == "")
        assert (MDP_1["lmc_weights_equil"] == MDP_1["weight_equil_wl_delta"] == "")
        assert MDP_2["init_wl_delta"] == 0.32
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP_1["init_lambda_weights"], [3.48, 2.78, 3.21, 4.56, 8.79, 0.48]
                )
            ]
        )
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP_2["init_lambda_weights"], [8.45, 0.52, 3.69, 2.43, 4.56, 6.73]
                )
            ]
        )

    def test_extract_final_dhdl_info(self, params_dict):
        EEXE = get_EEXE_instance(params_dict)
        dhdl_files = [
            os.path.join(input_path, f"dhdl/dhdl_{i}.xvg") for i in range(EEXE.n_sim)
        ]
        states, lambda_vecs = EEXE.extract_final_dhdl_info(dhdl_files)
        assert states == [5, 2, 2, 8]
        assert lambda_vecs == [(1, 0.25), (0.5, 0), (0.5, 0), (1, 1)]

    def test_extract_final_log_info(self, params_dict):
        EEXE = get_EEXE_instance(params_dict)
        log_files = [
            os.path.join(input_path, f"log/EXE_{i}.log") for i in range(EEXE.n_sim)
        ]
        wl_delta, weights, counts = EEXE.extract_final_log_info(log_files)
        assert wl_delta == [0.4, 0.5, 0.5, 0.5]
        assert np.allclose(weights, [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [0, 0.66431, 1.25475, 0.24443, 0.59472, 0.70726],
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701], ])
        assert counts == [
            [4, 11, 9, 9, 11, 6],
            [9, 8, 8, 11, 7, 7],
            [3, 1, 1, 9, 15, 21],
            [0, 0, 0, 1, 18, 31], ]
        assert EEXE.equil == [-1, -1, -1, -1]


"""
    def test_propose_swaps(self):
        random.seed(0)
        EEXE.n_sim = 4
        EEXE.state_ranges = [list(range(i, i + 5)) for i in range(EEXE.n_sim)]  # 5 states per replica
        states = [5, 2, 2, 7]   # This would lead to the swappables: [(0, 1), (0, 2), (1, 2)]

        # Case 1: Neighboring swapping (n_ex = 0 --> swappables = [(0, 1), (1, 2)])
        EEXE.n_ex = 0
        swap_list = EEXE.propose_swaps(states)
        assert swap_list == [(1, 2)]

        # Case 2: Multiple swaps (n_ex = 3)
        EEXE.n_ex = 5
        swap_list = EEXE.propose_swaps(states)
        assert swap_list == [(1, 2), (0, 2), (0, 1), (0, 2), (0, 2)]

        # Case 3: Empty swappable list
        states = [10, 10, 10, 10]
        swap_list = EEXE.propose_swaps(states)
        assert swap_list == []

    def test_get_swapped_configs(self):
        EEXE.state_ranges = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8]]
        states = [5, 2, 2, 8]
        lambda_vecs = [(1, 0.25), (0.5, 0), (0.5, 0), (1, 1)]
        weights = [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [0, 0.66431, 1.25475, 1.24443, 0.59472, 0.70726],   # the 4th prob was ajusted (from 0.24443) to tweak prob_acc  # noqa: E501
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701]]
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in range(4)]
        EEXE.mc_scheme = "metropolis"

        # Case 1: Empty swap list
        swap_list = []
        configs_1 = EEXE.get_swapping_pattern(swap_list, dhdl_files, states, lambda_vecs, weights)
        assert configs_1 == [0, 1, 2, 3]

        # Case 2: Multiple swaps
        swap_list = [(0, 2) for i in range(5)]   # prob_acc should be around 0.516
        random.seed(0)  # r1 = 0.844, r2 = 0.758, r3=0.421, r4=0.259 r5=0.511 --> 3 accepted moves --> [2, 1, 0, 3]
        configs_2 = EEXE.get_swapping_pattern(swap_list, dhdl_files, states, lambda_vecs, weights)
        assert configs_2 == [2, 1, 0, 3]

    def test_calc_prob_acc(self):
        EEXE.state_ranges = [
            [0, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8]]
        states = [5, 2, 2, 8]
        lambda_vecs = [(1, 0.25), (0.5, 0), (0.5, 0), (1, 1)]
        weights = [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [0, 0.66431, 1.25475, 0.24443, 0.59472, 0.70726],
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701]]
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in range(4)]

        # Test 1: Same-state swapping (True)
        swap = (1, 2)
        EEXE.mc_scheme = "same_state"
        prob_acc_1 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_1 == 1

        # Test 2: Same-state swapping (False)
        swap = (0, 2)
        prob_acc_2 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_2 == 0

        # Test 3: Metropolis-eq
        swap = (0, 2)
        EEXE.mc_scheme = "metropolis-eq"
        prob_acc_3 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_3 == 1    # Delta U = (-9.1366697 + 4.9963939)/2.478956208925815 ~ -1.67 kT

        # Test 4: Metropolis
        swap = (0, 2)
        EEXE.mc_scheme = "metropolis"
        prob_acc_4 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_4 == pytest.approx(0.18989559074633955)   # check this number again

    def test_accept_or_reject(self):
        random.seed(0)
        swap_bool_1 = EEXE.accept_or_reject(0)
        swap_bool_2 = EEXE.accept_or_reject(0.8)  # rand = 0.844
        swap_bool_3 = EEXE.accept_or_reject(0.8)  # rand = 0.758

        assert swap_bool_1 is False
        assert swap_bool_2 is False
        assert swap_bool_3 is True

    def test_historgam_correction(self):
        # Case 1: No histogram correction
        EEXE.N_cutoff = -1
        weights_1 = [[0, 10.304, 20.073, 29.364]]
        counts_1 = [[31415, 45701, 55457, 59557]]
        weights_1 = EEXE.histogram_correction(weights_1, counts_1)
        assert weights_1 == [[0, 10.304, 20.073, 29.364]]

        # Case 2: Perform histogram correction (N_cutoff reached)
        EEXE.verbose = False
        EEXE.N_cutoff = 5000
        weights_1 = EEXE.histogram_correction(weights_1, counts_1)
        assert np.allclose(weights_1, [
            [
                0,
                10.304 + np.log(31415 / 45701),
                20.073 + np.log(45701 / 55457),
                29.364 + np.log(55457 / 59557),
            ]
        ])  # noqa: E501

        # Case 3: Perform histogram correction (N_cutoff not reached)
        EEXE.verbose = True
        weights_2 = [[0, 10.304, 20.073, 29.364]]
        counts_2 = [[3141, 4570, 5545, 5955]]
        weights_2 = EEXE.histogram_correction(weights_2, counts_2)
        assert np.allclose(weights_2, [[0, 10.304, 20.073, 29.364 + np.log(5545 / 5955)]])

    def test_combine_weights(self):
        EEXE.n_tot = 6
        EEXE.n_sub = 4
        EEXE.s = 1
        EEXE.n_sim = 3
        EEXE.state_ranges = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
        weights = [[0, 2.1, 4.0, 3.7], [0, 1.7, 1.2, 2.6], [0, -0.4, 0.9, 1.9]]

        w1, g_vec_1 = EEXE.combine_weights(weights, method=None)
        w2, g_vec_2 = EEXE.combine_weights(weights, method='mean')

        EEXE.verbose = False   # just to reach some print statementss with verbose = False
        w3, g_vec_3 = EEXE.combine_weights(weights, method='geo-mean')
        w4, g_vec_4 = EEXE.combine_weights(weights, method='g-diff')

        assert np.allclose(w1, weights)
        assert np.allclose(w2, [
            [0.0, 2.20097, 3.99803, 3.59516],
            [0.0, 1.79706, 1.39419, 2.69607],
            [0.0, -0.40286, 0.89901, 1.88303]])
        assert np.allclose(w3, [
            [0.0, 2.2, 3.98889, 3.58889],
            [0.0, 1.78889, 1.38889, 2.68333],
            [0.0, -0.4, 0.89444, 1.87778]])
        assert np.allclose(w4, [
            [0, 2.1, 3.9, 3.5],
            [0, 1.8, 1.4, 2.75],
            [0, -0.4, 0.95, 1.95]])
        assert g_vec_1 is None
        assert np.allclose(list(g_vec_2), [0.0, 2.200968785917372, 3.9980269151210854, 3.5951633659351256, 4.897041830662871, 5.881054277773005])  # noqa: E501
        assert np.allclose(list(g_vec_3), [0.0, 2.1999999999999997, 3.9888888888888885, 3.5888888888888886, 4.883333333333334, 5.866666666666667])  # noqa: E501
        assert np.allclose(list(g_vec_4), [0, 2.1, 3.9, 3.5, 4.85, 5.85])

    def test_run_EEXE(self):
        # We probably can only test serial EEXE
        rank = MPI.COMM_WORLD.Get_rank()
        EEXE = EnsembleEXE('ensemble_md/tests/data/params.yaml')
        if rank == 0:
            for i in range(EEXE.n_sim):
                os.mkdir(f'sim_{i}')
                os.mkdir(f'sim_{i}/iteration_0')
                MDP = EEXE.initialize_MDP(i)
                MDP.write(f'sim_{i}/iteration_0/expanded.mdp', skipempty=True)
                shutil.copy('ensemble_md/tests/data/sys.gro', f'sim_{i}/iteration_0/sys.gro')
                shutil.copy('ensemble_md/tests/data/sys.top', f'sim_{i}/iteration_0/sys.top')

        md = EEXE.run_EEXE(0)   # just test the first iteration is fine
        assert md.output.returncode.result() == [0, 0, 0, 0]

        if rank == 0:
            os.system('rm -r sim_*')
            os.system('rm -r gmxapi.commandline.cli*_i0*')
"""
