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
import shutil
import pytest
import random
import numpy as np
import ensemble_md
import gmxapi as gmx
from mpi4py import MPI
from ensemble_md.ensemble_EXE import EnsembleEXE
from ensemble_md.exceptions import ParameterError

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")
EEXE = EnsembleEXE(os.path.join(input_path, "params.yaml"))


class Test_EnsembleEXE:
    def test_init_1(self):
        k = 1.380649e-23
        NA = 6.0221408e23
        assert EEXE.mc_scheme == "metropolis"
        assert EEXE.w_scheme is None
        assert EEXE.N_cutoff == 1000
        assert EEXE.n_ex == 0
        assert EEXE.output == "run_EEXE_log.txt"
        assert EEXE.mdp == "ensemble_md/tests/data/expanded.mdp"
        assert EEXE.gro == "ensemble_md/tests/data/sys.gro"
        assert EEXE.top == "ensemble_md/tests/data/sys.top"
        assert EEXE.nsteps == 500
        assert EEXE.dt == 0.002
        assert EEXE.temp == 298
        assert EEXE.kT == k * NA * 298 / 1000
        assert EEXE.n_tot == 9
        assert EEXE.n_sub == 6
        assert EEXE.runtime_args is None
        assert EEXE.state_ranges == [
            {0, 1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5, 6},
            {2, 3, 4, 5, 6, 7},
            {3, 4, 5, 6, 7, 8},
        ]
        assert EEXE.equil == [-1, -1, -1, -1]
        assert EEXE.nst_sim == 500
        assert EEXE.lambda_dict == {
            (0, 0): 0,
            (0.25, 0): 1,
            (0.5, 0): 2,
            (0.75, 0): 3,
            (1, 0): 4,
            (1, 0.25): 5,
            (1, 0.5): 6,
            (1, 0.75): 7,
            (1, 1): 8,
        }  # noqa: E501
        assert EEXE.lambda_ranges == [
            [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25)],
            [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5)],
            [(0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75)],
            [(0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0)],
        ]

    def test_init_2(self):
        yaml_0 = os.path.join(input_path, "other_yamls/0.yaml")
        with pytest.raises(ParameterError, match="The specified weight combining scheme is not available. Options include None, 'mean', and 'geo-mean'/'geo_mean'."):  # noqa: E501
            E0 = EnsembleEXE(yaml_0)  # noqa: F841

        yaml_1 = os.path.join(input_path, "other_yamls/1.yaml")
        with pytest.raises(ParameterError, match=f"Required parameter 'parallel' not specified in {yaml_1}"):
            E1 = EnsembleEXE(yaml_1)  # noqa: F841

        yaml_2 = os.path.join(input_path, "other_yamls/2.yaml")
        with pytest.raises(ParameterError, match="The specified MC scheme is not available. Options include 'same-state', 'metropolis', and 'metropolis-eq'."):  # noqa: E501
            E2 = EnsembleEXE(yaml_2)  # noqa: F841

        yaml_3 = os.path.join(input_path, "other_yamls/3.yaml")
        with pytest.raises(ParameterError, match="The parameter 'n_sim' should be an integer."):
            E3 = EnsembleEXE(yaml_3)  # noqa: F841

        yaml_4 = os.path.join(input_path, "other_yamls/4.yaml")
        with pytest.raises(ParameterError, match="The parameter 'n_iter' should be positive."):
            E4 = EnsembleEXE(yaml_4)  # noqa: F841

        yaml_5 = os.path.join(input_path, "other_yamls/5.yaml")
        with pytest.raises(ParameterError, match="The parameter 'n_ex' should be non-negative."):
            E5 = EnsembleEXE(yaml_5)  # noqa: F841

        yaml_6 = os.path.join(input_path, "other_yamls/6.yaml")
        with pytest.raises(ParameterError, match="The parameter 'N_cutoff' should be non-negative unless no histogram correction is needed, i.e. N_cutoff = -1."):  # noqa: E501
            E6 = EnsembleEXE(yaml_6)  # noqa: F841

        yaml_7 = os.path.join(input_path, "other_yamls/7.yaml")
        with pytest.raises(ParameterError, match="The parameter 'mdp' should be a string."):
            E7 = EnsembleEXE(yaml_7)  # noqa: F841

        yaml_8 = os.path.join(input_path, "other_yamls/8.yaml")
        with pytest.raises(ParameterError, match="The parameter 'parallel' should be a boolean variable."):
            E8 = EnsembleEXE(yaml_8)  # noqa: F841

        yaml_9 = os.path.join(input_path, "other_yamls/9.yaml")
        E9 = EnsembleEXE(yaml_9)
        assert E9.lambda_dict == {
            (0.1,): 0,
            (0.2,): 1,
            (0.3,): 2,
            (0.4,): 3,
            (0.5,): 4,
            (0.6,): 5,
            (0.7,): 6,
            (0.8,): 7,
            (1,): 8,
        }  # noqa: E501
        assert E9.lambda_ranges == [
            [(0.1,), (0.2,), (0.3,), (0.4,), (0.5,), (0.6,)],
            [(0.2,), (0.3,), (0.4,), (0.5,), (0.6,), (0.7,)],
            [(0.3,), (0.4,), (0.5,), (0.6,), (0.7,), (0.8,)],
            [(0.4,), (0.5,), (0.6,), (0.7,), (0.8,), (1.0,)],
        ]

        yaml_10 = os.path.join(input_path, "other_yamls/10.yaml")
        E10 = EnsembleEXE(yaml_10)
        assert E10.lambda_dict == {
            (0, 0, 0): 0,
            (0.25, 0, 0): 1,
            (0.5, 0, 0): 2,
            (0.75, 0, 0): 3,
            (1, 0, 0): 4,
            (1, 0.25, 0.2): 5,
            (1, 0.5, 0.6): 6,
            (1, 0.75, 0.8): 7,
            (1, 1, 1): 8,
        }  # noqa: E501
        assert E10.lambda_ranges == [
            [(0.0, 0.0, 0.0), (0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.25, 0.2)],
            [(0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.25, 0.2), (1.0, 0.5, 0.6)],
            [(0.5, 0.0, 0.0), (0.75, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.25, 0.2), (1.0, 0.5, 0.6), (1.0, 0.75, 0.8)],
            [(0.75, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.25, 0.2), (1.0, 0.5, 0.6), (1.0, 0.75, 0.8), (1.0, 1.0, 1.0)],
        ]
        assert E10.runtime_args == {'-nt': '16', '-ntomp': 8}

    def test_print_params(self, capfd):
        # capfd is a fixture in pytest for testing STDOUT
        EEXE.print_params()
        out, err = capfd.readouterr()
        L = ""
        L += "\nImportant parameters of EXEE\n============================\n"
        L += f"gmxapi version: {gmx.__version__}\nensemble_md version: {ensemble_md.__version__}\n"
        L += "Output log file: run_EEXE_log.txt\nVerbose log file: True\nWhether the replicas run in parallel: False\n"
        L += "MC scheme for swapping simulations: metropolis\nScheme for combining weights: None\n"
        L += "Histogram cutoff: 1000\nNumber of replicas: 4\nNumber of iterations: 10\n"
        L += "Number of exchanges in one attempt: 0\n"
        L += "Length of each replica: 1.0 ps\nTotal number of states: 9\n"
        L += "Additional runtime arguments: None\n"
        L += "Alchemical ranges of each replica in EEXE:\n  - Replica 0: States [0, 1, 2, 3, 4, 5]\n"
        L += "  - Replica 1: States [1, 2, 3, 4, 5, 6]\n  - Replica 2: States [2, 3, 4, 5, 6, 7]\n"
        L += "  - Replica 3: States [3, 4, 5, 6, 7, 8]\n"
        assert out == L

    def test_initialize_MDP(self):
        MDP = EEXE.initialize_MDP(2)
        assert MDP["nsteps"] == 500
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP["vdw-lambdas"], [0.00, 0.00, 0.00, 0.25, 0.50, 0.75]
                )
            ]
        )
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP["coul-lambdas"], [0.50, 0.75, 1.00, 1.00, 1.00, 1.00]
                )
            ]
        )
        assert all(
            [a == b for a, b in zip(MDP["init-lambda-weights"], [0, 0, 0, 0, 0, 0])]
        )

    def test_update_MDP(self):
        new_template = "ensemble_md/tests/data/expanded.mdp"
        iter_idx = 3
        states = [2, 5, 7, 4]
        wl_delta = [0.4, 0.32, 0.256, 0.32]
        weights = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [3.48, 2.78, 3.21, 4.56, 8.79, 0.48],
            [8.45, 0.52, 3.69, 2.43, 4.56, 6.73],
        ]  # noqa: E501
        EEXE.equil = [-1, 35, 0, -1]
        MDP_1 = EEXE.update_MDP(
            new_template, 2, iter_idx, states, wl_delta, weights)
        MDP_2 = EEXE.update_MDP(
            new_template, 3, iter_idx, states, wl_delta, weights)

        assert MDP_1["tinit"] == MDP_2["tinit"] == 3
        assert MDP_1["nsteps"] == MDP_2["nsteps"] == 500
        assert MDP_1["init-lambda-state"] == 5
        assert MDP_2["init-lambda-state"] == 1
        assert (
            MDP_1["init-wl-delta"] == MDP_1["wl-scale"] == MDP_1["wl-ratio"] == ""
        )  # because equil_bools is True
        assert (
            MDP_1["lmc-weights-equil"] == MDP_1["weight-equil-wl-delta"] == ""
        )  # because equil_bools is True
        assert MDP_2["init-wl-delta"] == 0.32
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP_1["init-lambda-weights"], [3.48, 2.78, 3.21, 4.56, 8.79, 0.48]
                )
            ]
        )
        assert all(
            [
                a == b
                for a, b in zip(
                    MDP_2["init-lambda-weights"], [8.45, 0.52, 3.69, 2.43, 4.56, 6.73]
                )
            ]
        )

    def test_extract_final_dhdl_info(self):
        dhdl_files = [
            os.path.join(input_path, f"dhdl_{i}.xvg") for i in range(EEXE.n_sim)
        ]
        states, lambda_vecs = EEXE.extract_final_dhdl_info(dhdl_files)
        assert states == [5, 2, 2, 8]
        assert lambda_vecs == [(1, 0.25), (0.5, 0), (0.5, 0), (1, 1)]

    def test_extract_final_log_info(self):
        EEXE.equil = [-1, -1, -1, -1]
        log_files = [
            os.path.join(input_path, f"EXE_{i}.log") for i in range(EEXE.n_sim)
        ]
        wl_delta, weights, counts = EEXE.extract_final_log_info(log_files)
        assert wl_delta == [0.4, 0.5, 0.5, 0.5]
        assert np.allclose(weights, [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [0, 0.66431, 1.25475, 0.24443, 0.59472, 0.70726],
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701],
        ])
        assert counts == [
            [4, 11, 9, 9, 11, 6],
            [9, 8, 8, 11, 7, 7],
            [3, 1, 1, 9, 15, 21],
            [0, 0, 0, 1, 18, 31],
        ]
        assert EEXE.equil == [-1, -1, -1, -1]

    def test_propose_swaps(self):
        random.seed(0)
        EEXE.n_sim = 4
        EEXE.state_ranges = [set(range(i, i + 5)) for i in range(EEXE.n_sim)]  # 5 states per replica
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
        assert swap_list is None

    def test_gest_swapped_configus(self):
        EEXE.state_ranges = [
            {0, 1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5, 6},
            {2, 3, 4, 5, 6, 7},
            {3, 4, 5, 6, 7, 8}]
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
            {0, 1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5, 6},
            {2, 3, 4, 5, 6, 7},
            {3, 4, 5, 6, 7, 8}]
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
        assert prob_acc_3 == 1    # \Delta U = (-9.1366697 + 4.9963939)/2.478956208925815 ~ -1.67 kT

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
        EEXE.state_ranges = [{0, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 4, 5}]
        weights = [[0, 2.1, 4.0, 3.7], [0, 1.7, 1.2, 2.6], [0, -0.4, 0.9, 1.9]]

        w1, g_vec_1 = EEXE.combine_weights(weights, method=None)
        w2, g_vec_2 = EEXE.combine_weights(weights, method='mean')

        EEXE.verbose = False   # just to reach some print statementss with verbose = False
        w3, g_vec_3 = EEXE.combine_weights(weights, method='geo-mean')

        assert np.allclose(w1, weights)
        assert np.allclose(w2, [
            [0.0, 2.20097, 3.99803, 3.59516],
            [0.0, 1.79706, 1.39419, 2.69607],
            [0.0, -0.40286, 0.89901, 1.88303]])
        assert np.allclose(w3, [
            [0.0, 2.2, 3.98889, 3.58889],
            [0.0, 1.78889, 1.38889, 2.68333],
            [0.0, -0.4, 0.89444, 1.87778]])
        assert g_vec_1 is None
        assert np.allclose(list(g_vec_2), [0.0, 2.200968785917372, 3.9980269151210854, 3.5951633659351256, 4.897041830662871, 5.881054277773005])  # noqa: E501
        assert np.allclose(list(g_vec_3), [0.0, 2.1999999999999997, 3.9888888888888885, 3.5888888888888886, 4.883333333333334, 5.866666666666667])  # noqa: E501

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
            os.system('rm -r gmxapi.commandline.cli1_i0*')
