import os
import random
import numpy as np
import ensemble_md
import ensemble_md.gmx_parser as gmx_parser
from ensemble_md.ensemble_EXE import EnsembleEXE

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")
EEXE = EnsembleEXE(os.path.join(input_path, "params.yaml"))


class Test_EnsembleEXE:
    def test_init(self):
        k = 1.380649e-23
        NA = 6.0221408e23
        assert EEXE.mc_scheme == "metropolis"
        assert EEXE.w_scheme == "exp-avg"
        assert EEXE.N_cutoff == 1000
        assert EEXE.n_pairs == 1
        assert EEXE.outfile == "results.txt"
        assert EEXE.mdp == "data/expanded.mdp"
        assert EEXE.nsteps == 500
        assert EEXE.dt == 0.002
        assert EEXE.temp == 298
        assert EEXE.kT == k * NA * 298 / 1000
        assert EEXE.n_tot == 9
        assert EEXE.n_sub == 6
        assert EEXE.state_ranges == [
            {0, 1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5, 6},
            {2, 3, 4, 5, 6, 7},
            {3, 4, 5, 6, 7, 8},
        ]
        assert EEXE.nst_sim == 500

    def test_print_params(self, capfd):
        # capfd is a fixture in pytest for testing STDOUT
        EEXE.print_params()
        out, err = capfd.readouterr()
        L = ""
        L += "\nImportant parameters of EXEE\n============================\n"
        L += f"gmxapi version: 0.3.2\nensemble_md version: {ensemble_md.__version__}\n"
        L += "Output log file: results.txt\nWhether the replicas run in parallel: False\n"
        L += "MC scheme for swapping simulations: metropolis\nScheme for combining weights: exp-avg\n"
        L += "Histogram cutoff: 1000\nNumber of replicas: 4\nNumber of iterations: 10\n"
        L += "Length of each replica: 1.0 ps\nTotal number of states: 9\n"
        L += "States sampled by each simulation:\n  - Simulation 0: States [0, 1, 2, 3, 4, 5]\n"
        L += "  - Simulation 1: States [1, 2, 3, 4, 5, 6]\n  - Simulation 2: States [2, 3, 4, 5, 6, 7]\n"
        L += "  - Simulation 3: States [3, 4, 5, 6, 7, 8]\n"
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
        new_template = gmx_parser.MDP("data/expanded.mdp")
        iter_idx = 3
        states = [2, 5, 7, 4]
        wl_delta = [0.4, 0.32, 0.256, 0.32]
        weights = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [3.48, 2.78, 3.21, 4.56, 8.79, 0.48],
            [8.45, 0.52, 3.69, 2.43, 4.56, 6.73],
        ]  # noqa: E501
        equil_bools = [False, True, True, False]
        MDP_1 = EEXE.update_MDP(
            new_template, 2, iter_idx, states, wl_delta, weights, equil_bools
        )
        MDP_2 = EEXE.update_MDP(
            new_template, 3, iter_idx, states, wl_delta, weights, equil_bools
        )

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

    def test_map_lambda2state(self):
        EEXE.map_lambda2state()
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

    def test_extract_final_dhdl_info(self):
        dhdl_files = [
            os.path.join(input_path, f"dhdl_{i}.xvg") for i in range(EEXE.n_sim)
        ]
        states, lambda_vecs = EEXE.extract_final_dhdl_info(dhdl_files)
        assert states == [5, 2, 2, 8]
        assert lambda_vecs == [(1, 0.25), (0.5, 0), (0.5, 0), (1, 1)]

    def test_extract_final_log_info(self):
        log_files = [
            os.path.join(input_path, f"EXE_{i}.log") for i in range(EEXE.n_sim)
        ]
        wl_delta, weights, counts, equil_bools = EEXE.extract_final_log_info(log_files)
        assert wl_delta == [0.4, 0.5, 0.5, 0.5]
        assert weights == [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [0, 0.66431, 1.25475, 0.24443, 0.59472, 0.70726],
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701],
        ]
        assert counts == [
            [4, 11, 9, 9, 11, 6],
            [9, 8, 8, 11, 7, 7],
            [3, 1, 1, 9, 15, 21],
            [0, 0, 0, 1, 18, 31],
        ]
        assert equil_bools == [False, False, False, False]

    def test_propose_swaps(self):
        random.seed(0)
        EEXE.n_sim = 8
        EEXE.n_pairs = 5
        EEXE.state_ranges = [
            set(range(i, i + 5)) for i in range(EEXE.n_sim)
        ]  # 12 states, 5 for each replica
        swap_list = EEXE.propose_swaps()
        assert EEXE.n_pairs == 4
        assert swap_list == [
            (3, 4),
            (5, 6),
            (0, 1),
        ]  # The remaining pair of (2, 7) is not swappable

    def test_calc_prob_acc(self):
        EEXE.state_ranges = [
            {0, 1, 2, 3, 4, 5},
            {1, 2, 3, 4, 5, 6},
            {2, 3, 4, 5, 6, 7},
            {3, 4, 5, 6, 7, 8},
        ]
        states = [5, 2, 2, 8]
        lambda_vecs = [(1, 0.25), (0.5, 0), (0.5, 0), (1, 1)]
        weights = [
            [0, 1.03101, 2.55736, 3.63808, 4.47220, 6.13408],
            [0, 1.22635, 2.30707, 2.44120, 4.10308, 6.03106],
            [
                0,
                0.66431,
                1.25475,
                -5.24443,
                0.59472,
                0.70726,
            ],  # Changed the 4th from 0.24443 to -5.24443
            [0, 0.09620, 1.59937, -4.31679, -22.89436, -28.08701],
        ]

        # Test 1: Swapping states not present in both lambda ranges
        swap = (0, 3)
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in swap]
        prob_acc_1 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_1 == 0

        # Test 2: Same-state swapping (True)
        swap = (1, 2)
        EEXE.mc_scheme = "same_state"
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in swap]
        prob_acc_2 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_2 == 1

        # Test 3: Same-state swapping (False)
        swap = (0, 2)
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in swap]
        prob_acc_3 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_3 == 0

        # Test 4: Metropolis-eq
        EEXE.mc_scheme = "metropolis-eq"
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in swap]
        prob_acc_4 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_4 == 1

        # Test 5: Metropolis
        EEXE.mc_scheme = "metropolis"
        dhdl_files = [os.path.join(input_path, f"dhdl_{i}.xvg") for i in swap]
        prob_acc_5 = EEXE.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
        assert prob_acc_5 is None
        # assert prob_acc_5 == 0.13207042981597653  # check this number again

    def test_accept_or_reject(self):
        random.seed(0)
        swap_bool_1 = EEXE.accept_or_reject(0)
        swap_bool_2 = EEXE.accept_or_reject(0.8)  # rand = 0.844
        swap_bool_3 = EEXE.accept_or_reject(0.8)  # rand = 0.758

        assert swap_bool_1 is False
        assert swap_bool_2 is False
        assert swap_bool_3 is True

    def test_historgam_correction(self):
        EEXE.N_cutoff = 5000
        weights_1 = [[0, 10.304, 20.073, 29.364]]
        counts_1 = [[31415, 45701, 55457, 59557]]
        weights_1 = EEXE.histogram_correction(weights_1, counts_1)
        assert weights_1 == [
            [
                0,
                10.304 + np.log(31415 / 45701),
                20.073 + np.log(45701 / 55457),
                29.364 + np.log(55457 / 59557),
            ]
        ]  # noqa: E501

        weights_2 = [[0, 10.304, 20.073, 29.364]]
        counts_2 = [[3141, 4570, 5545, 5955]]
        weights_2 = EEXE.histogram_correction(weights_2, counts_2)
        assert weights_2 == [[0, 10.304, 20.073, 29.364 + np.log(5545 / 5955)]]

    def combine_w_inputs(self):
        swap = (0, 1)
        weights = [[0, 2.1, 4.0, 3.7, 4.8], [0, -0.4, 0.7, 1.5, 2.4]]
        counts = [[31, 29, 13, 48, 21], [21, 27, 36, 19, 15]]  # will not be usd though
        return weights, counts, swap

    def test_combine_weights(self):
        EEXE.state_ranges = [{0, 1, 2, 3, 4}, {2, 3, 4, 5, 6}]

        EEXE.w_scheme = None
        w1 = np.array(EEXE.combine_weights(*self.combine_w_inputs()))
        np.testing.assert_array_almost_equal(
            w1, np.array([[0, 2.1, 4.0, 3.7, 4.8], [0, -0.4, 0.7, 1.5, 2.4]])
        )

        EEXE.w_scheme = "avg"
        w2 = np.array(EEXE.combine_weights(*self.combine_w_inputs()))
        np.testing.assert_array_almost_equal(
            w2, np.array([[0, 2.1, 4.0, 3.65, 4.75], [0, -0.35, 0.75, 1.5, 2.4]])
        )

        EEXE.w_scheme = "exp-avg"
        w3 = np.array(EEXE.combine_weights(*self.combine_w_inputs()))
        np.testing.assert_array_almost_equal(
            w3,
            np.array(
                [
                    [
                        0,
                        2.1,
                        4.0,
                        -np.log(0.5 * (np.exp(-3.7) + np.exp(-3.6))),
                        -np.log(0.5 * (np.exp(-4.8) + np.exp(-4.7))),
                    ],
                    [
                        0,
                        -np.log(0.5 * (np.exp(0.4) + np.exp(0.3))),
                        -np.log(0.5 * (np.exp(-0.7) + np.exp(-0.8))),
                        1.5,
                        2.4,
                    ],
                ]
            ),
        )

        EEXE.w_scheme = "hist-exp-avg"  # should be the same as exp-avg because of low histogram counts
        w4 = np.array(EEXE.combine_weights(*self.combine_w_inputs()))
        np.testing.assert_array_almost_equal(
            w4,
            np.array(
                [
                    [
                        0,
                        2.1,
                        4.0,
                        -np.log(0.5 * (np.exp(-3.7) + np.exp(-3.6))),
                        -np.log(0.5 * (np.exp(-4.8) + np.exp(-4.7))),
                    ],
                    [
                        0,
                        -np.log(0.5 * (np.exp(0.4) + np.exp(0.3))),
                        -np.log(0.5 * (np.exp(-0.7) + np.exp(-0.8))),
                        1.5,
                        2.4,
                    ],
                ]
            ),
        )

    def test_run_EEXE(self):
        pass
