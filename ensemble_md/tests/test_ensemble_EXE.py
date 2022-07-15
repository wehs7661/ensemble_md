import os
import numpy as np 
import ensemble_md
import ensemble_md.gmx_parser as gmx_parser
from ensemble_md.ensemble_EXE import EnsembleEXE

np.random.seed(0)
current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, 'data')
EEXE = EnsembleEXE(os.path.join(input_path, 'params.yaml'))


class Test_EnsembleEXE:
    def test_init(self):
        k = 1.380649E-23
        NA = 6.0221408E+23
        assert EEXE.mc_scheme == 'metropolis'
        assert EEXE.w_scheme == 'exp-avg'
        assert EEXE.N_cutoff == 1000
        assert EEXE.n_pairs == 1
        assert EEXE.outfile == 'results.txt' 
        assert EEXE.mdp == 'data/expanded.mdp'
        assert EEXE.nsteps == 500
        assert EEXE.dt == 0.002 
        assert EEXE.temp == 298
        assert EEXE.kT == k * NA * 298 / 1000
        assert EEXE.n_tot == 9
        assert EEXE.n_sub == 6
        assert EEXE.state_ranges == [{0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 6}, {2, 3, 4, 5, 6, 7}, {3, 4, 5, 6, 7, 8}]
        assert EEXE.nst_sim == 500

    def test_print_params(self, capfd):
        # capfd is a fixture in pytest for testing STDOUT
        EEXE.print_params()
        out, err = capfd.readouterr()
        L = ''
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
        assert all([a == b for a, b in zip(MDP["vdw-lambdas"], [0.00, 0.00, 0.00, 0.25, 0.50, 0.75])])
        assert all([a == b for a, b in zip(MDP["coul-lambdas"], [0.50, 0.75, 1.00, 1.00, 1.00, 1.00])])
        assert all([a == b for a, b in zip(MDP["init-lambda-weights"], [0, 0, 0, 0, 0, 0])])

    def test_update_MDP(self):
        new_template = gmx_parser.MDP('data/expanded.mdp')
        iter_idx = 3
        states = [2, 5, 7, 4]
        wl_delta = [0.4, 0.32, 0.256, 0.32]
        weights = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [3.48, 2.78, 3.21, 4.56, 8.79, 0.48], [8.45, 0.52, 3.69, 2.43, 4.56, 6.73]]
        equil_bools = [False, True, True, False]
        MDP_1 = EEXE.update_MDP(new_template, 2, iter_idx, states, wl_delta, weights, equil_bools)
        MDP_2 = EEXE.update_MDP(new_template, 3, iter_idx, states, wl_delta, weights, equil_bools)
        
        assert MDP_1["tinit"] == MDP_2['tinit'] == 3
        assert MDP_1["nsteps"] == MDP_2['nsteps'] == 500
        assert MDP_1['init-lambda-state'] == 5
        assert MDP_2['init-lambda-state'] == 1
        assert MDP_1['init-wl-delta'] == MDP_1['wl-scale'] == MDP_1['wl-ratio']  == ''  # because equil_bools is True
        assert MDP_1['lmc-weights-equil'] == MDP_1['weight-equil-wl-delta'] == ''  # because equil_bools is True
        assert MDP_2['init-wl-delta'] == 0.32
        assert all([a == b for a, b in zip(MDP_1['init-lambda-weights'], [3.48, 2.78, 3.21, 4.56, 8.79, 0.48])])
        assert all([a == b for a, b in zip(MDP_2['init-lambda-weights'], [8.45, 0.52, 3.69, 2.43, 4.56, 6.73])])

    def test_map_lambda2state(self):
        EEXE.map_lambda2state()
        assert EEXE.lambda_dict == {(0, 0): 0, (0.25, 0): 1, (0.5, 0): 2, (0.75, 0): 3, (1, 0): 4, (1, 0.25): 5, (1, 0.5): 6, (1, 0.75): 7, (1, 1): 8}
        assert EEXE.lambda_ranges == [
            [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25)],
            [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5)],
            [(0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75)],
            [(0.75, 0.0), (1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75), (1.0, 1.0)]
        ]

    def test_extract_final_dhdl_info(self):
        pass

    def test_extract_final_log_info(self):
        pass 

    def test_propose_swaps(self):
        pass

    def test_calc_prob_acc(self):
        pass

    def test_accept_or_reject(self):
        pass

def test_historgam_correction():
    pass 

def test_run_EEXE():
    pass


