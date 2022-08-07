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
The :code:`ensemble_EXE` module helps set up ensembles of expanded ensemble.
"""
import os
import copy
import yaml
import random
import numpy as np
from mpi4py import MPI
from itertools import combinations
from alchemlyb.parsing.gmx import extract_dHdl
from alchemlyb.parsing.gmx import _get_headers as get_headers
from alchemlyb.parsing.gmx import _extract_dataframe as extract_dataframe

import gmxapi as gmx
import ensemble_md
import ensemble_md.gmx_parser as gmx_parser
import ensemble_md.utils as utils
from ensemble_md.exceptions import ParameterError


rank = MPI.COMM_WORLD.Get_rank()  # Note that this is a GLOBAL variable


class EnsembleEXE:
    """
    This class helps set up input files of an ensemble of expanded ensemble.
    """

    def __init__(self, yml_file):
        """
        Sets up or reads in the user-defined parameters from the yaml file and the mdp template.

        Parameters
        ----------
        yml_file : str
            The file name of the YAML file for specifying the parameters for EEXE.
        outfile : str
            The file name of the log file for documenting how different replicas interact
            during the process.
        """
        # Step 0: Set up constants
        k = 1.380649e-23
        NA = 6.0221408e23

        # Step 1: Read in parameters from the YAML file.
        with open(yml_file) as f:
            try:
                params = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                params = yaml.load(f)

        for attr in params:
            setattr(self, attr, params[attr])

        # Step 2: Handle the YAML parameters
        required_args = ["parallel", "n_sim", "n_iterations", "s", "mdp"]
        for i in required_args:
            if hasattr(self, i) is False:
                raise ParameterError(
                    f"Required parameter '{i}' not specified in {yml_file}."
                )  # noqa: F405

        # Key: Optional argument; Value: Default value
        optional_args = {
            "mc_scheme": "metropolis",
            "w_scheme": "exp-avg",
            "N_cutoff": 1000,
            "n_ex": 0,       # neighbor swaps
            "outfile": "results.txt",
            "verbose": True
        }
        for i in optional_args:
            if hasattr(self, i) is False:
                setattr(self, i, optional_args[i])

        # Step 3: Check if the parameters in the YAML file is well-defined
        if self.mc_scheme not in ['same-state', 'same_state', 'metropolis', 'metropolis-eq', 'metropolis_eq']:
            raise ParameterError("The specified MC scheme is not available. Options include 'same-state', 'metropolis', and 'metropolis-eq'.")  # noqa: E501

        params_int = ['n_sim', 'n_iterations', 's', 'nst_sim', 'N_cutoff', 'n_ex']  # integer parameters
        for i in params_int:
            if type(getattr(self, i)) != int:
                raise ParameterError(f"The parameter {i} should be an integer.")

        params_pos = ['n_sim', 'n_iterations', 's', 'nst_sim']  # positive parameters
        for i in params_pos:
            if getattr(self, i) <= 0:
                raise ParameterError(f"The parameter {i} should be positive.")

        params_non_neg = ['N_cutoff', 'n_ex']  # non-negative parameters
        for i in params_non_neg:
            if getattr(self, i) < 0:
                raise ParameterError(f"The parameter {i} should be non-negative.")

        params_str = ['mdp', 'outfile']
        for i in params_str:
            if type(getattr(self, i)) != str:
                raise ParameterError(f"The parameter {i} should be a string.")

        params_bool = ['parallel', 'verbose']
        for i in params_bool:
            if type(getattr(self, i)) != bool:
                raise ParameterError(f"The parameter {i} should be a boolean variable.")

        # Step 4: Read in parameters from the MDP template
        self.template = gmx_parser.MDP(self.mdp)
        self.nsteps = self.template["nsteps"]  # will be overwritten by self.nst_sim if nst_sim is specified.
        self.dt = self.template["dt"]  # ps
        self.temp = self.template["ref_t"]
        self.kT = k * NA * self.temp / 1000  # 1 kT in kJ/mol

        # Total # of states. n_tot = n_sub * n_sim - (n_overlap) * (n_sum - 1), where n_overlap = n_sub - s
        self.n_tot = len(self.template["vdw-lambdas"])

        # Number of states of each replica (assuming the same for all rep)
        self.n_sub = self.n_tot - self.s * (self.n_sim - 1)

        # A list of sets of state indices
        self.state_ranges = [
            set(range(i * self.s, i * self.s + self.n_sub)) for i in range(self.n_sim)
        ]

        if hasattr(self, "nst_sim") is False:
            self.nst_sim = self.nsteps

        # Map the lamda vectors to state indices
        self.map_lambda2state()

    def print_params(self):
        """
        Prints out important parameters
        """
        if rank == 0:
            print("\nImportant parameters of EXEE")
            print("============================")
            print(f"gmxapi version: {gmx.__version__}")
            print(f"ensemble_md version: {ensemble_md.__version__}")
            print(f"Output log file: {self.outfile}")
            print(f"Verbose log file: {self.verbose}")
            print(f"Whether the replicas run in parallel: {self.parallel}")
            print(f"MC scheme for swapping simulations: {self.mc_scheme}")
            print(f"Scheme for combining weights: {self.w_scheme}")
            print(f"Histogram cutoff: {self.N_cutoff}")
            print(f"Number of replicas: {self.n_sim}")
            print(f"Number of iterations: {self.n_iterations}")
            print(f"Number of exchanges in one attempt: {self.n_ex}")
            print(f"Length of each replica: {self.dt * self.nst_sim} ps")
            print(f"Total number of states: {self.n_tot}")
            print("States sampled by each simulation/replica:")
            for i in range(self.n_sim):
                print(f"  - Simulation {i}: States {list(self.state_ranges[i])}")

    def map_lambda2state(self):
        """
        Returns a dictionary whose keys are vectors of coupling
        parameters and values are the corresponding state indices (starting from 0).

        Attributes
        ----------
        lambda_dict : dict
            A dictionary whose keys are tuples of coupling parameters and
            values are the corresponding GLOBAL state indices (starting from 0).
        lambda_ranges : list
            A list of lambda vectors of the state range of each replica.
        """
        self.lambda_dict = {}  # key: vector of coupling parameters, value: state index
        for i in range(self.n_tot):
            # Note the order of the lambda values in the vector is the same as the dataframe generated by extract_dhdl
            if "coul-lambdas" in self.template:
                if "restraint-lambdas" in self.template:
                    self.lambda_dict[
                        (
                            self.template["coul-lambdas"][i],
                            self.template["vdw-lambdas"][i],
                            self.template["restraint_lambdas"][i],
                        )
                    ] = i
                else:
                    self.lambda_dict[
                        (
                            self.template["coul-lambdas"][i],
                            self.template["vdw-lambdas"][i],
                        )
                    ] = i
            else:
                self.lambda_dict[(self.template["vdw-lambdas"][i])] = i

        self.lambda_ranges = [[list(self.lambda_dict.keys())[j] for j in self.state_ranges[i]]for i in range(len(self.state_ranges))]  # noqa: E501

    def initialize_MDP(self, idx):
        """
        Initializes the MDP object for generating MDP files for a replica based on the MDP template.
        Note that this is only for generating MDP files for the FIRST iteration and it has nothing
        to do with whether the weights are fixed or equilibrating. The user needs to make sure that
        the MDP template has all the common parameters of all replicas.

        Parameters
        ----------
        idx : int
            The index of the simulation whose MDP parameters need to be initialized.

        Returns
        -------
        MDP : gmx_parser.MDP obj
            An updated object of gmx_parser.MDP that can be used to write MDP files.
        """
        MDP = copy.deepcopy(self.template)
        MDP["nsteps"] = self.nst_sim
        MDP["vdw-lambdas"] = self.template["vdw-lambdas"][idx * self.s:idx * self.s + self.n_sub]
        if "coul-lambdas" in self.template:
            coul = self.template["coul-lambdas"]
            MDP["coul-lambdas"] = coul[idx * self.s: idx * self.s + self.n_sub]
        if "init-lambda-weights" in self.template:
            init_w = self.template["init-lambda-weights"]
            MDP["init-lambda-weights"] = init_w[idx: idx + self.n_sub]

        return MDP

    def update_MDP(self, new_template, sim_idx, iter_idx, states, wl_delta, weights, equil_bools):
        """
        Updates the MDP file for a new iteration based on the new MDP template coming from the previous iteration.
        Note that if the weights got equilibrated in the previous iteration, then we need to fix the weights in
        later iterations.

        Parameters
        ----------
        new_template : gmx_parser.MDP obj
            The gmx_parser.MDP object of the new MDP template. Typically the MDP file of the previous iteration.
        sim_idx : int
            The index of the simulation whose MDP parameters need to be updated.
        iter_idx : int
            The index of the iteration to be performed later.
        states : list
            A list of last sampled states of all simulaitons in the previous iteration.
        wl_delta : list
            A list of final Wang-Landau incrementors of all simulations.
        weights : list
            A list of lists final weights of all simulations.
        equil_bools : list
            A list of booleans indicating if the weights of the simulations have been equilibrated.

        Return
        ------
        MDP : gmx_parser.MDP obj
            An updated object of gmx_parser.MDP that can be used to write MDP files.
        """
        MDP = copy.deepcopy(new_template)
        MDP["tinit"] = self.nst_sim * self.dt * iter_idx
        MDP["nsteps"] = self.nst_sim
        MDP["init-lambda-state"] = (states[sim_idx] - sim_idx * self.s)  # 2nd term is for shifting from the global to local index.  # noqa: E501
        MDP["init-lambda-weights"] = weights[sim_idx]
        MDP["init-wl-delta"] = wl_delta[sim_idx]

        if equil_bools[sim_idx] is False:
            MDP["init-wl-delta"] = wl_delta[sim_idx]
        else:
            MDP["lmc-stats"] = "no"
            MDP["wl-scale"] = ""
            MDP["wl-ratio"] = ""
            MDP["init-wl-delta"] = ""
            MDP["lmc-weights-equil"] = ""
            MDP["weight-equil-wl-delta"] = ""

        return MDP

    def extract_final_dhdl_info(self, dhdl_files):
        """
        For all the replica simulations, this function finds the last sampled state
        and the corresponding lambda values from a dhdl file.

        Parameters
        ----------
        dhdl_files : list
            A list of dhdl file names

        Returns
        -------
        states : list
            A list of the global indices of the last sampled states of all simulaitons.
        lambda_vecs : list
            A list of lambda vectors corresponding to the last sampled states of all simulations.
        """
        states, lambda_vecs = [], []
        if self.verbose is True:
            print('\n', end='')
        print("Below are the final states being visited:")
        for j in range(self.n_sim):
            dhdl = extract_dHdl(dhdl_files[j], T=self.temp)
            lambda_vecs.append(dhdl.index[-1][1:])
            states.append(self.lambda_dict[lambda_vecs[-1]])  # absolute order
            print(
                f"  Simulation {j}: State {states[j]}, (coul, vdw) = \
                {list(self.lambda_dict.keys())[list(self.lambda_dict.values()).index(states[j])]}"
            )
        print('\n', end='')

        return states, lambda_vecs

    def extract_final_log_info(self, log_files):
        """
        For all the replica simulations, this function finds the following information from a log file.

          - The final Wang-Landau incrementors.
          - The final lists of weights.
          - The final lists of counts.
          - Whether the weights were equilibrated in the simulations.

        Parameters
        ----------
        log_files : list
            A list of log file names

        Returns
        -------
        wl_delta : list
            A list of final Wang-Landau incrementors of all simulations.
        weights : list
            A list of lists of final weights of all simulations.
        counts : list
            A list of lists of final counts of all simulations.
        equil_bools : list
            A list of booleans indicating if the weights were equilibrated in the simulation.
        """
        wl_delta, weights, counts, equil_bools = [], [], [], []

        # 2. Find the final Wang-Landau incrementors and weights
        for j in range(self.n_sim):
            result = gmx_parser.parse_log(log_files[j])
            wl_delta.append(result[0])
            weights.append(result[1])
            counts.append(result[2])
            equil_bools.append(result[3])

        return wl_delta, weights, counts, equil_bools

    def propose_swaps(self, states):
        """
        Proposes swaps of coordinates between replicas by drawing samples from the swappable pairs,
        which are defined as pairs of simulations whose last sampled states are in the alchemical ranges
        of both simulations. This is required, or ΔH and Δg will be unknown. Note that this automatically
        assume the simulations to be swapped should have overlapping lambda ranges.

        Parameters
        ----------
        states : list
            A list of last sampled states of ALL simulaitons. Typically generated by extract_final_dhdl_info.

        Returns
        -------
        swap_list : list
            A list of tuples of simulation indices to be swapped.
        """
        swap_list = []
        sim_idx = list(range(self.n_sim))

        # Before drawing samples, we need to identify the swappable pairs
        all_pairs = list(combinations(sim_idx, 2))

        # First, we identify pairs of replicas with overlapping ranges
        swappables = [i for i in all_pairs if self.state_ranges[i[0]].intersection(self.state_ranges[i[1]]) != set()]  # noqa: E501
        print(f"Replicas with overlapping λ ranges: {swappables}")

        # Then, from these pairs, we exclude the ones whose the last sampled states are not present in both alchemical ranges  # noqa: E501
        # In this case, U^i_n, U_^j_m, g^i_n, and g_^j_m are unknown and the probability cannot be calculated.
        swappables = [i for i in swappables if states[i[0]] in self.state_ranges[i[1]] and states[i[1]] in self.state_ranges[i[0]]]  # noqa: E501

        if self.n_ex == 0:
            n_ex = 1    # One swap will be carried out.
            print('Note: At most only 1 swap will be carried out, which is between neighboring replicas.')
            swappables = [i for i in swappables if np.abs(i[0] - i[1]) == 1]
        else:
            n_ex = self.n_ex

        print(f"Swappable pairs: {swappables}")

        for i in range(n_ex):
            try:
                swap_list = random.choices(swappables, k=n_ex)
            except IndexError:
                # In the case that swappables is an empty list, i.e. no swappable pairs.
                swap_list = None

        return swap_list

    def get_swapped_configs(self, swap_list, dhdl_files, states, lambda_vecs, weights):
        """
        Finds the configuration each replica corresponds to after multiple swaps proposed in one attempt.

        Parameters
        ----------
        swap_list : list or None
            A list of tuples of simulation indices to be swapped.
        dhdl_files : list
            A list of dhdl files of ALL simulations. Note that the filename should be ordered
            with ascending simulation/replica indices, i.e. the n-th filename in the list should be
            the dhdl file of the n-th simulation.
        states : list
            A list of last sampled states of ALL simulaitons. Typically generated by extract_final_dhdl_info.
        lambda_vecs : list
            A list of lambda vectors corresponding to the last sampled states of ALL simulations.
            Typically generated by extract_final_dhdl_info.
        weights : list
            A list of lists of final weights of ALL simulations. Typiecally generated by extract_final_log_info.

        Returns
        -------
        configs : list
            The list of the final configurations after all the swaps.

        """
        configs = list(range(self.n_sim))   # Can be regarded as the indices corresponding to dhdl files/configurations
        if swap_list is None:
            print('No swap is proposed because there is no swappable pair at all.')
        else:
            for i in range(len(swap_list)):
                swap = swap_list[i]
                if self.verbose is True:
                    print(f'\nA swap ({i + 1}/{len(swap_list)}) is proposed between Simulation {swap[0]} (state {states[swap[0]]}) and Simulation {swap[1]} (state {states[swap[1]]}) ...')  # noqa: E501

                # For each swap, calculate the acceptance ratio and decide whether to accept the swap.
                prob_acc = self.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
                swap_bool = self.accept_or_reject(prob_acc)

                # Each dhdl file corresponds to one configuration, so if the swap is accepted,
                # we switch the order of the two corresponding dhdl file names in the dhdl_files.
                # Also, the indices in configs need to be update correspondingly.
                if swap_bool is True:
                    # The assignments need to be done at the same time in just one line.
                    dhdl_files[swap[0]], dhdl_files[swap[1]] = dhdl_files[swap[1]], dhdl_files[swap[0]]
                    configs[swap[0]], configs[swap[1]] = configs[swap[1]], configs[swap[0]]
                else:
                    pass

                if self.verbose is True:
                    print(f'(Current configurations: {configs})')
                else:
                    if i == len(swap_list) - 1:
                        print(f'\n{len(swap_list)} swaps have been proposed.')
                        print(f'Final configuration: {configs}')

        return configs

    def calc_prob_acc(self, swap, dhdl_files, states, lambda_vecs, weights):
        """
        Calculates the acceptance ratio given the MC scheme for swapping the simulations.

        Parameters
        ----------
        swap : tuple
            A tuple of indices corresponding to the simulations to be swapped.
        dhdl_files : list
            A list of dhdl files of ALL simulations. Note that the filename should be ordered
            with ascending simulation/replica indices, i.e. the n-th filename in the list should be
            the dhdl file of the n-th simulation.
        states : list
            A list of last sampled states of ALL simulaitons. Typically generated by extract_final_dhdl_info.
        lambda_vecs : list
            A list of lambda vectors corresponding to the last sampled states of ALL simulations.
            Typically generated by extract_final_dhdl_info.
        weights : list
            A list of lists of final weights of ALL simulations. Typiecally generated by extract_final_log_info.

        Returns
        -------
        prob_acc : float
            The acceptance ratio
        """
        if self.mc_scheme == "same_state" or self.mc_scheme == "same-state":
            if states[swap[0]] == states[swap[1]]:  # same state, swap!
                prob_acc = 1  # This must lead to an output of swap_bool = True from the function accept_or_reject
            else:
                prob_acc = 0  # This must lead to an output of swap_bool = False from the function accept_or_reject

        else:  # i.e. metropolis-eq or metropolis, which both require the calculation of dU
            # Now we calculate dU
            if self.verbose is True:
                print("  Proposing a move from (x^i_m, x^j_n) to (x^i_n, x^j_m) ...")
            f0, f1 = dhdl_files[swap[0]], dhdl_files[swap[1]]
            h0, h1 = get_headers(f0), get_headers(f1)
            data_0, data_1 = (
                extract_dataframe(f0, headers=h0).iloc[-1],
                extract_dataframe(f1, headers=h1).iloc[-1],
            )

            # \Delta H to all states at the last time frame
            # Notably, the can be regarded as H for each state since the reference state will have a value of 0 anyway.
            dhdl_0 = data_0[-self.n_sub:]
            dhdl_1 = data_1[-self.n_sub:]

            new_lambda_0 = lambda_vecs[swap[1]]  # new lambda vector (tuple) for swap[0]
            new_lambda_1 = lambda_vecs[swap[0]]  # new lambda vector (tuple) for swap[1]

            # old local index, will only be used in "metropolis"
            old_state_0 = int(data_0[1])
            old_state_1 = int(data_1[1])

            new_state_0 = self.lambda_ranges[swap[0]].index(new_lambda_0)  # new state index (local index in simulation swap[0]) # noqa: E501
            new_state_1 = self.lambda_ranges[swap[1]].index(new_lambda_1)  # new state index (local index in simulation swap[1]) # noqa: E501

            dU_0 = (dhdl_0[new_state_0] - dhdl_0[old_state_0]) / self.kT  # U^{i}_{n} - U^{i}_{m}, i.e. \Delta U (kT) to the new state  # noqa: E501
            dU_1 = (dhdl_1[new_state_1] - dhdl_1[old_state_1]) / self.kT  # U^{j}_{m} - U^{j}_{n}, i.e. \Delta U (kT) to the new state  # noqa: E501
            dU = dU_0 + dU_1
            if self.verbose is True:
                print(
                    f"  U^i_n - U^i_m = {dU_0:.2f} kT, U^j_m - U^j_n = {dU_1:.2f} kT, Total dU: {dU:.2f} kT"
                )

            if self.mc_scheme == "metropolis_eq" or self.mc_scheme == "metropolis-eq":
                prob_acc = min(1, np.exp(-dU))
            else:  # must be 'metropolis', which consider lambda weights as well
                g0 = weights[swap[0]]
                g1 = weights[swap[1]]
                dg_0 = g0[new_state_0] - g0[old_state_0]  # g^{i}_{n} - g^{i}_{m}
                dg_1 = g1[new_state_1] - g1[old_state_1]  # g^{j}_{m} - g^{j}_{n}
                dg = dg_0 + dg_1  # kT

                """
                Note that simulations with different lambda ranges would have different references
                so g^{i}_{n} - g^{j}_{n} or g^{j}_{m} - g^{i}_{m} wouldn't not make sense.
                We therefore print g^i_n - g^i_m and g^j_m - g^j_n instead even if they are less interesting.
                """
                if self.verbose is True:
                    print(
                        f"  g^i_n - g^i_m = {dg_0:.2f} kT, g^j_m - g^j_n = {dg_1:.2f} kT, Total dg: {dg:.2f} kT"
                    )

                prob_acc = min(1, np.exp(-dU + dg))
        return prob_acc

    def accept_or_reject(self, prob_acc):
        """
        Returns a boolean variable indiciating whether the proposed swap should be acceepted given the acceptance rate.

        Parameters
        ----------
        prob_acc : float
            The acceptance rate.

        Returns
        -------
        swap_bool : bool
            A boolean variable indicating whether the swap should be accepted.
        """
        if prob_acc == 0:
            swap_bool = False
            if self.verbose is True:
                print("  Swap rejected! ")
        else:
            rand = random.random()
            if self.verbose is True:
                print(
                    f"  Acceptance rate: {prob_acc:.3f} / Random number drawn: {rand:.3f}"
                )
            if rand < prob_acc:
                swap_bool = True
                # Below we flush the buffer so the next STDOUT (current configs) will be appended
                if self.verbose is True:
                    print("  Swap accepeted! ", end="", flush=True)
            else:
                swap_bool = False
                if self.verbose is True:
                    print("  Swap rejected! ", end="", flush=True)

        return swap_bool

    def histogram_correction(self, weights, counts):
        """
        Corrects the lambda weights based on the histogram counts. Namely,
        :math:`g_k' = g_k + ln(N_{k-1}/N_k)`, where :math:`g_k` and :math:`g_k'`
        are the lambda weight after and before the correction, respectively.
        Notably, in any of the following situations, we don't do any correction.

        - Either :math:`N_{k-1}` or :math:`N_k` is 0.
        - Either :math:`N_{k-1}` or :math:`N_k` is smaller than the histogram cutoff.

        Parameters
        ----------
        weights : list
            A list of lists of weights (of ALL simulations) to be corrected.
        counts : list
            A list of lists of counts (of ALL simulations).

        Return
        ------
        weights : list
            An updated list of lists of corected weights.
        """
        print("\nPerforming histogram correction for the lambda weights ...")
        for i in range(len(weights)):  # loop over the replicas
            print(f"  Counts of rep {i}:\t\t{counts[i]}")
            print(
                f'  Original weights of rep {i}:\t{[float(f"{k:.3f}") for k in weights[i]]}'
            )
            for j in range(1, len(weights[i])):  # loop over the alchemical states
                if counts[i][j - 1] != 0 and counts[i][j - 1] != 0:
                    if np.min([counts[i][j - 1], counts[i][j]]) > self.N_cutoff:
                        weights[i][j] += np.log(counts[i][j - 1] / counts[i][j])
            print(
                f'  Corrected weights of rep {i}:\t{[float(f"{k:.3f}") for k in weights[i]]}\n'
            )
        return weights

    def combine_weights(self, weights, counts, swap):
        """
        Combines the lambda weights of the exchanging replicas as needed.

        Parameters
        ----------
        weights : list
            A list of Wang Landau weights of all simulations.
        counts : list
            A list of final counts of all simulations.
        swap : tuple
            A tuple of a pair of simulation indices to be swapped.

        Returns
        -------
        weights : list
            A list of updated Wang Landau weights of all simulations. Note that the weights
            of the simulations not involved in the exchange will remain the same.
        """
        if self.w_scheme is None:
            pass
        else:
            # Step 1: "Correct" the weights as needed (hist-exp-avg or mbar-exp-avg).
            # If min(N_{k-1}, N_k) < cutoff, there's no correction and hist-exp-avg/mbar-exp-avg reduces to exp-avg.
            if self.w_scheme == "hist-exp-avg":
                weights = self.histogram_correction(weights, counts)
            elif self.w_scheme == "mbar-exp-avg":
                pass

            # Step 2: Combine the weights
            print(
                f"Performing weight combination between simulation {swap[0]} and simulation {swap[1]} ... "
            )
            overlap = self.state_ranges[swap[0]].intersection(
                self.state_ranges[swap[1]]
            )
            print(
                f"  Alchemical range of simulation {swap[0]}: {list(self.state_ranges[swap[0]])}"
            )
            print(
                f"  Alchemical range of simulation {swap[1]}: {list(self.state_ranges[swap[1]])}"
            )
            print(f"  Overlapped alchemical ranges: {list(overlap)}\n")

            # swap[0] is always smaller than swap[1]
            g_0 = np.array(weights[swap[0]][-len(overlap):])  # the last N values
            g_1 = np.array(
                weights[swap[1]][: len(overlap)]
            )  # the first N values (g_1[0] must be 0)
            shifted_0 = g_0 - g_0[0]  # will be combined with g_1 to generate modified_1
            shifted_1 = g_1 + g_0[0]  # will be combined with g_0 to generate modified_0
            print(f'  Original g^i: {[float(f"{i:.3f}") for i in weights[swap[0]]]}')
            print(f'  Original g^j: {[float(f"{i:.3f}") for i in weights[swap[1]]]}')

            if self.w_scheme == "avg":
                # DEPRECATED! Kept here just for conveneint testing.
                modified_0 = (shifted_1 + g_0) / 2
                modified_1 = (shifted_0 + g_1) / 2
                weights[swap[0]][-len(overlap):] = modified_0
                weights[swap[1]][: len(overlap)] = modified_1
            else:
                # This includes exp-avg, hist-exp-avg or mbar-exp-avg
                modified_0 = -np.log((np.exp(-g_0) + np.exp(-shifted_1)) / 2)
                modified_1 = -np.log((np.exp(-g_1) + np.exp(-shifted_0)) / 2)
                weights[swap[0]][-len(overlap):] = modified_0
                weights[swap[1]][: len(overlap)] = modified_1

            print(f'  Modified g^i: {[float(f"{i:.3f}") for i in weights[swap[0]]]}')
            print(f'  Modified g^j: {[float(f"{i:.3f}") for i in weights[swap[1]]]}')

        return weights

    def run_EEXE(self, n):
        """
        Makes tpr files and run an ensemble of expanded ensemble simulations
        using :code:`gmxapi.mdrun`.

        Parameters
        ----------
        n : int
            The iteration index (starting from 0).
        """
        if rank == 0:
            iter_str = f'\nIteration {n}: {self.dt * self.nst_sim * n: .1f} - {self.dt * self.nst_sim * (n + 1): .1f} ps'  # noqa: E501
            print(iter_str + '\n' + '=' * (len(iter_str) - 1))

        if rank == 0:
            dir_before = [
                i for i in os.listdir(".") if os.path.isdir(os.path.join(".", i))
            ]
            if self.verbose is True:
                print("Preparing the tpr files for the simulation ensemble...")

        grompp = gmx.commandline_operation(
            "gmx",
            arguments=["grompp"],  # noqa: E127
            input_files=[  # noqa: E127
                {
                    "-f": f"../sim_{i}/iteration_{n}/expanded.mdp",
                    "-c": f"../sim_{i}/iteration_{n}/sys.gro",
                    "-p": f"../sim_{i}/iteration_{n}/sys.top",
                }
                for i in range(self.n_sim)
            ],
            output_files=[  # noqa: E127
                {  # noqa: E127
                    "-o": f"../sim_{i}/iteration_{n}/sys_EE.tpr",
                    "-po": f"../sim_{i}/iteration_{n}/mdout.mdp",
                }
                for i in range(self.n_sim)
            ],
        )
        grompp.run()
        if rank == 0:  # just print the messages once
            utils.gmx_output(grompp, self.verbose)

        # Run all the simulations simultaneously using gmxapi
        if rank == 0 and self.verbose is True:
            print("Running an ensemble of simulations ...")

        if self.parallel is True:
            tpr = [f'{grompp.output.file["-o"].result()[i]}' for i in range(self.n_sim)]
            inputs = gmx.read_tpr(tpr)
            md = gmx.mdrun(inputs)
            md.run()
        else:
            # Note that we could use output_files argument to customize the output file
            # names but here we'll just use the defaults.
            md = gmx.commandline_operation(
                "gmx",
                arguments=["mdrun"],  # noqa: E128
                input_files=[  # noqa: E128
                    {
                        "-s": grompp.output.file["-o"].result()[i],
                    }
                    for i in range(self.n_sim)
                ],
            )
            md.run()
            if rank == 0:  # just print the messages once
                utils.gmx_output(md, self.verbose)

        if rank == 0:
            dir_after = [
                i for i in os.listdir(".") if os.path.isdir(os.path.join(".", i))
            ]
            utils.clean_up(dir_before, dir_after, self.verbose)

        return md
