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
The :obj:`.ensemble_EXE` module provides functions for setting up and ensemble of expanded ensemble.
"""
import os
import sys
import copy
import yaml
import shutil
import random
import numpy as np
from mpi4py import MPI
from itertools import combinations
from collections import OrderedDict
from alchemlyb.parsing.gmx import extract_dHdl
from alchemlyb.parsing.gmx import _get_headers as get_headers
from alchemlyb.parsing.gmx import _extract_dataframe as extract_dataframe

import gmxapi as gmx
import ensemble_md
from ensemble_md.utils import utils
from ensemble_md.utils import gmx_parser
from ensemble_md.utils.exceptions import ParameterError


rank = MPI.COMM_WORLD.Get_rank()  # Note that this is a GLOBAL variable


class EnsembleEXE:
    """
    This class provides a variety of functions useful for setting up and running
    an ensemble of expanded ensemble.
    """

    def __init__(self, yaml_file):
        self.yaml = yaml_file
        self.set_params()

    def set_params(self):
        """
        Sets up or reads in the user-defined parameters from a yaml file and an MDP template.
        This function is called to instantiate the class in the :code:`__init__` function of
        class. Specifically, it does the following:

          1. Sets up constants.
          2. Reads in parameters from a YAML file.
          3. Handles YAML parameters.
          4. Checks if the parameters in the YAML file are well-defined.
          5. Reformats the input MDP file to replace all hyphens with underscores.
          6. Reads in parameters from the MDP template.

        After instantiation, the class instance will have attributes corresponding to
        each of the parameters specified in the YAML file. For a full list of the parameters that
        can be specified in the YAML file, please refer to :ref:`doc_parameters`.

        :param yaml_file: The file name of the YAML file for specifying the parameters for EEXE.
        :type yaml_file: str

        :raises ParameterError:

              - If a required parameter is not specified in the YAML file.
              - If a specified parameter is not recognizable.
              - If a specified weight combining scheme is not available.
              - If a specified MC scheme is not available.
              - If a specified free energy estimator is not available.
              - If a specified method for error estimation is not available.
              - If an integer parameter is not an integer.
              - If a positive parameter is not positive.
              - If a non-negative parameter is negative.
              - If an invalid MDP file is detected.
        """
        self.warnings = []  # Store warnings, if any.

        # Step 0: Set up constants
        k = 1.380649e-23
        NA = 6.0221408e23

        # Step 1: Read in parameters from the YAML file.
        with open(self.yaml) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        for attr in params:
            setattr(self, attr, params[attr])

        # Step 2: Handle the YAML parameters
        required_args = [
            "gro",
            "top",
            "mdp",
            "parallel",
            "n_sim",
            "n_iter",
            "s",
        ]
        for i in required_args:
            if hasattr(self, i) is False or getattr(self, i) is None:
                raise ParameterError(
                    f"Required parameter '{i}' not specified in {self.yaml}."
                )  # noqa: F405

        # Key: Optional argument; Value: Default value
        optional_args = {
            "nst_sim": None,
            "mc_scheme": "metropolis",
            "w_scheme": None,
            "N_cutoff": 1000,
            "n_ex": 0,       # neighbor swaps
            "verbose": True,
            "runtime_args": None,
            "n_ckpt": 100,
            "msm": False,
            "free_energy": False,
            "df_spacing": 1,
            "df_method": "MBAR",
            "err_method": "propagate",
            "n_bootstrap": 50,
            "seed": None,
        }
        for i in optional_args:
            if hasattr(self, i) is False or getattr(self, i) is None:
                setattr(self, i, optional_args[i])

        # all_args: Arguments that can be specified in the YAML file.
        all_args = required_args + list(optional_args.keys())
        for i in params:
            if i not in all_args:
                self.warnings.append(f'Warning: Parameter "{i}" specified in the input YAML file is not recognizable.')

        # Step 3: Check if the parameters in the YAML file are well-defined
        if self.w_scheme not in [None, 'mean', 'geo-mean', 'g-diff']:
            raise ParameterError("The specified weight combining scheme is not available. Available options include None, 'mean', 'geo-mean'/'geo_mean' and 'g-diff/g_diff'.")  # noqa: E501

        if self.mc_scheme not in [None, 'same-state', 'same_state', 'metropolis', 'metropolis-eq', 'metropolis_eq']:
            raise ParameterError("The specified MC scheme is not available. Available options include 'same-state', 'metropolis', and 'metropolis-eq'.")  # noqa: E501

        if self.df_method not in [None, 'TI', 'BAR', 'MBAR']:
            raise ParameterError("The specified free energy estimator is not available. Available options include 'TI', 'BAR', and 'MBAR'.")  # noqa: E501

        if self.err_method not in [None, 'propagate', 'bootstrap']:
            raise ParameterError("The specified method for error estimation is not available. Available options include 'propagate', and 'bootstrap'.")  # noqa: E501

        params_int = ['n_sim', 'n_iter', 's', 'nst_sim', 'N_cutoff', 'df_spacing', 'n_ckpt', 'n_bootstrap']  # integer parameters  # noqa: E501
        if self.n_ex != 'N^3':
            params_int.append('n_ex')
        if self.seed is not None:
            params_int.append('seed')
        for i in params_int:
            if type(getattr(self, i)) != int:
                raise ParameterError(f"The parameter '{i}' should be an integer.")

        params_pos = ['n_sim', 'n_iter', 's', 'nst_sim', 'n_ckpt', 'df_spacing', 'n_bootstrap']  # positive parameters
        for i in params_pos:
            if getattr(self, i) <= 0:
                raise ParameterError(f"The parameter '{i}' should be positive.")

        if self.n_ex != 'N^3' and self.n_ex < 0:
            raise ParameterError("The parameter 'n_ex' should be non-negative.")

        if self.N_cutoff < 0 and self.N_cutoff != -1:
            raise ParameterError("The parameter 'N_cutoff' should be non-negative unless no histogram correction is needed, i.e. N_cutoff = -1.")  # noqa: E501

        params_str = ['gro', 'top', 'mdp']
        for i in params_str:
            if type(getattr(self, i)) != str:
                raise ParameterError(f"The parameter '{i}' should be a string.")

        params_bool = ['parallel', 'verbose', 'msm', 'free_energy']
        for i in params_bool:
            if type(getattr(self, i)) != bool:
                raise ParameterError(f"The parameter '{i}' should be a boolean variable.")

        # Step 4: Reformat the input MDP file to replace all hypens with underscores.
        self.reformat_MDP()

        # Step 5: Read in parameters from the MDP template
        self.template = gmx_parser.MDP(self.mdp)
        self.nsteps = self.template["nsteps"]  # will be overwritten by self.nst_sim if nst_sim is specified.
        self.dt = self.template["dt"]  # ps
        self.temp = self.template["ref_t"]
        self.kT = k * NA * self.temp / 1000  # 1 kT in kJ/mol

        if 'wl_scale' in self.template.keys():
            if self.template['wl_scale'] != '':
                self.fixed_weights = False
            else:
                self.fixed_weights = True
        else:
            self.fixed_weights = True

        if 'lmc_seed' in self.template and self.template['lmc_seed'] != -1:
            self.warnings.append('Warning: We recommend setting lmc_seed as -1 so the random seed is different for each iteration.')  # noqa: E501

        if 'gen_seed' in self.template and self.template['gen_seed'] != -1:
            self.warnings.append('Warning: We recommend setting gen_seed as -1 so the random seed is different for each iteration.')  # noqa: E501

        if 'symmetrized_transition_matrix' in self.template and self.template['symmetrized_transition_matrix'] == 'yes':  # noqa: E501
            self.warnings.append('Warning: We recommend setting symmetrized-transition-matrix to no instead of yes.')

        if self.template['nstlog'] > self.nst_sim:
            raise ParameterError(
                'The parameter "nstlog" should be equal to or smaller than "nst_sim" specified in the YAML file so that the sampling information can be parsed.')  # noqa: E501

        # Step 6: Set up derived parameters
        # 6-1. Total # of states: n_tot = n_sub * n_sim - (n_overlap) * (n_sim - 1), where n_overlap = n_sub - s
        self.n_tot = len(self.template["vdw_lambdas"])

        # 6-2. Number of states of each replica (assuming the same for all rep)
        self.n_sub = self.n_tot - self.s * (self.n_sim - 1)
        if self.n_sub < 1:
            raise ParameterError(
                f"There must be at least two states for each replica (current value: {self.n_sub}). The current specified configuration (n_tot={self.n_tot}, n_sim={self.n_sim}, s={self.s}) does not work for EEXE.")  # noqa: E501

        # 6-3. A list of sets of state indices
        start_idx = [i * self.s for i in range(self.n_sim)]
        self.state_ranges = [list(np.arange(i, i + self.n_sub)) for i in start_idx]

        # 6-4. A list of simulation statuses to be updated
        self.equil = [-1 for i in range(self.n_sim)]   # -1 means unequilibrated

        # 6-5. Numbe of steps per iteration
        if self.nst_sim is None:
            self.nst_sim = self.nsteps

        # 6-6. Map the lamda vectors to state indices
        self.map_lambda2state()

        # 6-7. For counting the number swap attempts and the rejected ones
        self.n_rejected = 0
        self.n_swap_attempts = 0

        # 6-8. Replica space trajectories. For example, rep_trajs[0] = [0, 2, 3, 0, 1, ...] means
        # that configuration 0 transitioned to replica 2, then 3, 0, 1, in iterations 1, 2, 3, 4, ...,
        # respectively. The first element of rep_traj[i] should always be i.
        self.rep_trajs = [[i] for i in range(self.n_sim)]

        # 6-9. The time series of the (processed) whole-range alchemical weights
        # If no weight combination is applied, self.g_vecs will just be a list of None's.
        self.g_vecs = []

        # 6-10. Data analysis
        if self.df_method == 'MBAR':
            self.get_u_nk = True
            self.get_dHdl = False
        else:
            self.get_u_nk = False
            self.get_dHdl = True

    def print_params(self, params_analysis=False):
        """
        Prints important parameters related to the EXEE simulation.

        Parameters
        ----------
        params_analysis : bool, optional
            If True, additional parameters related to data analysis will be printed. Default is False.
        """
        print("Important parameters of EXEE")
        print("============================")
        print(f"Python version: {sys.version}")
        print(f"gmxapi version: {gmx.__version__}")
        print(f"ensemble_md version: {ensemble_md.__version__}")
        print(f'Simulation inputs: {self.gro}, {self.top}, {self.mdp}')
        print(f"Verbose log file: {self.verbose}")
        print(f"Whether the replicas run in parallel: {self.parallel}")
        print(f"MC scheme for swapping simulations: {self.mc_scheme}")
        print(f"Scheme for combining weights: {self.w_scheme}")
        print(f"Histogram cutoff: {self.N_cutoff}")
        print(f"Number of replicas: {self.n_sim}")
        print(f"Number of iterations: {self.n_iter}")
        print(f"Number of exchanges in one attempt: {self.n_ex}")
        print(f"Length of each replica: {self.dt * self.nst_sim} ps")
        print(f"Frequency for checkpointing: {self.n_ckpt} iterations")
        print(f"Total number of states: {self.n_tot}")
        print(f"Additional runtime arguments: {self.runtime_args}")
        print("Alchemical ranges of each replica in EEXE:")
        for i in range(self.n_sim):
            print(f"  - Replica {i}: States {self.state_ranges[i]}")

        if params_analysis is True:
            print()
            print(f"Whether to build Markov state models and perform relevant analysis: {self.msm}")
            print(f"Whether to perform free energy calculations: {self.free_energy}")
            print(f"The step to used in subsampling the DHDL data in free energy calculations, if any: {self.df_spacing}")  # noqa: E501
            print(f"The chosen free energy estimator for free energy calculations, if any: {self.df_method}")
            print(f"The method for estimating the uncertainty of free energies in free energy calculations, if any: {self.err_method}")  # noqa: E501
            print(f"The number of bootstrap iterations in the boostrapping method, if used: {self.n_bootstrap}")
            print(f"The random seed to use in bootstrapping, if used: {self.seed}")

        if self.reformatted_mdp is True:
            print('Note that the input MDP file has been reformatted by replacing hypens with underscores. The original mdp file has been renamed as *backup.mdp.')  # noqa: E501

    def reformat_MDP(self):
        """
        Reformats an MDP file so that all hyphens in the parameter names are replaced by underscores.
        If the input MDP file contains hyphens in its parameter names, the class attribue :code:`self.reformatted`
        will be set to :code:`True`. In this case, the new MDP object with reformatted parameter names will be
        written to the original file paththe file, while the original file will be renamed with a
        :code:`_backup` suffix. If the input MDP file contains underscores in its parameter names, the function sets
        the class attribute :code:`self.reformatted_mdp` to :code:`False`.
        """
        params = gmx_parser.MDP(self.mdp)

        odict = OrderedDict([(k.replace('-', '_'), v) for k, v in params.items()])
        params_new = gmx_parser.MDP(None, **odict)

        if params_new.keys() == params.keys():
            self.reformatted_mdp = False  # no need to reformat the file
        else:
            self.reformatted_mdp = True
            new_name = self.mdp.split('.mdp')[0] + '_backup.mdp'
            shutil.move(self.mdp, new_name)
            params_new.write(self.mdp)

    def map_lambda2state(self):
        """
        Returns a dictionary whose keys are vectors of coupling
        parameters and values are the corresponding state indices (starting from 0).

        Attributes
        ----------
        lambda_types : list
            A list of lambda types specified in the MDP file.
        lambda_dict : dict
            A dictionary whose keys are tuples of coupling parameters and
            values are the corresponding GLOBAL state indices (starting from 0).
        lambda_ranges : list
            A list of lambda vectors of the state range of each replica.
        """
        # A list of all possible lambda types in the order read by GROMACS, which is likely also the order when being printed to the log file.  # noqa: E501
        # See https://gitlab.com/gromacs/gromacs/-/blob/main/src/gromacs/gmxpreprocess/readir.cpp#L2543
        lambdas_types_all = ['fep_lambdas', 'mass_lambdas', 'coul_lambdas', 'vdw_lambdas', 'bonded_lambdas', 'restraint_lambdas', 'temperature_lambdas']  # noqa: E501
        self.lambda_types = []  # lambdas specified in the MDP file
        for i in lambdas_types_all:
            if i in self.template.keys():  # there shouldn't be parameters like "fep-lambdas" after reformatting the MDP file  # noqa: E501
                self.lambda_types.append(i)

        self.lambda_dict = {}  # key: vector of coupling parameters, value: state index
        for i in range(self.n_tot):
            key = tuple([self.template[j][i] for j in self.lambda_types])
            self.lambda_dict[key] = i
        self.lambda_ranges = [[list(self.lambda_dict.keys())[j] for j in self.state_ranges[i]]for i in range(len(self.state_ranges))]  # noqa: E501

    def initialize_MDP(self, idx):
        """
        Initializes the MDP object for generating MDP files for a replica based on the MDP template.
        This function should be called only for generating MDP files for the FIRST iteration
        and it has nothing to do with whether the weights are fixed or equilibrating.
        It is assumed that the MDP template has all the common parameters of all replicas.

        Parameters
        ----------
        idx : int
            Index of the simulation whose MDP parameters need to be initialized.

        Returns
        -------
        MDP : :obj:`.gmx_parser.MDP` obj
            An updated object of :obj:`.gmx_parser.MDP` that can be used to write MDP files.
        """
        MDP = copy.deepcopy(self.template)
        MDP["nsteps"] = self.nst_sim

        start_idx = idx * self.s
        for i in self.lambda_types:
            MDP[i] = self.template[i][start_idx:start_idx + self.n_sub]

        if "init_lambda_weights" in self.template:
            MDP["init_lambda_weights"] = self.template["init_lambda_weights"][start_idx:start_idx + self.n_sub]

        return MDP

    def update_MDP(self, new_template, sim_idx, iter_idx, states, wl_delta, weights):
        """
        Updates the MDP file for a new iteration based on the new MDP template coming from the previous iteration.
        Note that if the weights got equilibrated in the previous iteration, then the weights will be fixed
        at these equilibrated values for all the following iterations.

        Parameters
        ----------
        new_template : str
            The new MDP template file. Typically the MDP file of the previous iteration.
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

        Return
        ------
        MDP : :obj:`.gmx_parser.MDP` obj
            An updated object of :obj:`.gmx_parser.MDP` that can be used to write MDP files.
        """
        new_template = gmx_parser.MDP(new_template)  # turn into a gmx_parser.MDP object
        MDP = copy.deepcopy(new_template)
        MDP["tinit"] = self.nst_sim * self.dt * iter_idx
        MDP["nsteps"] = self.nst_sim
        MDP["init_lambda_state"] = (states[sim_idx] - sim_idx * self.s)  # 2nd term is for shifting from the global to local index.  # noqa: E501
        MDP["init_lambda_weights"] = weights[sim_idx]
        MDP["init_wl_delta"] = wl_delta[sim_idx]

        if self.equil[sim_idx] == -1:   # the weights haven't been equilibrated
            MDP["init_wl_delta"] = wl_delta[sim_idx]
        else:
            MDP["lmc_stats"] = "no"
            MDP["wl_scale"] = ""
            MDP["wl_ratio"] = ""
            MDP["init_wl_delta"] = ""
            MDP["lmc_weights_equil"] = ""
            MDP["weight_equil_wl_delta"] = ""

        return MDP

    def extract_final_dhdl_info(self, dhdl_files):
        """
        For all the replica simulations, finds the last sampled state
        and the corresponding lambda values from a dhdl file.

        Parameters
        ----------
        dhdl_files : list
            A list of GROMACS DHDL file names.

        Returns
        -------
        states : list
            A list of the global indices of the last sampled states of all simulaitons.
        lambda_vecs : list
            A list of lambda vectors corresponding to the last sampled states of all simulations.
        """
        states, lambda_vecs = [], []
        print("\nBelow are the final states being visited:")
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
        Extracts the following information for each replica simulation from a given list of GROMACS LOG files:

          - The final Wang-Landau incrementors.
          - The final lists of weights.
          - The final lists of counts.
          - Whether the weights were equilibrated in the simulations.

        Parameters
        ----------
        log_files : list
            A list of file paths to GROMACS LOG files.

        Returns
        -------
        wl_delta : list
            A list of final Wang-Landau incrementors of all simulations.
        weights : list
            A list of lists of final weights of all simulations.
        counts : list
            A list of lists of final counts of all simulations.
        """
        wl_delta, weights, counts = [], [], []

        # 2. Find the final Wang-Landau incrementors and weights
        for j in range(self.n_sim):
            if self.verbose:
                print(f'Parsing {log_files[j]} ...')
            result = gmx_parser.parse_log(log_files[j])
            wl_delta.append(result[0])
            weights.append(result[1])
            counts.append(result[2])

            # In Case 3, result[3] will be 0 but it will never be passed to self.equil[j]
            # because once self.equil[j] is not -1, we stop updating. This way, we can keep
            # the time when the weights get equilibrated all the way.
            if self.equil[j] == -1:
                self.equil[j] = result[3]
            else:
                pass

        return wl_delta, weights, counts

    def propose_swaps(self, states):
        """
        Proposes swaps of coordinates between replicas by drawing samples from the swappable pairs.
        A pair of simulations is considered swappable if their last sampled states are in the alchemical
        ranges of both simulations. This is required, as otherwise the values of ΔH and Δg will be unknown.
        Note that this assumes that the simulations to be swapped have overlapping lambda ranges.

        Parameters
        ----------
        states : list
            A list of the global indices of the last sampled states of all simulations. This is typically
            generated by the :obj:`.extract_final_dhdl_info` method.

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
        swappables = [i for i in all_pairs if set(self.state_ranges[i[0]]).intersection(set(self.state_ranges[i[1]])) != set()]  # noqa: E501
        print(f"\nReplicas with overlapping λ ranges: {swappables}")

        # Then, from these pairs, we exclude the ones whose the last sampled states are not present in both alchemical ranges  # noqa: E501
        # In this case, U^i_n, U_^j_m, g^i_n, and g_^j_m are unknown and the probability cannot be calculated.
        swappables = [i for i in swappables if states[i[0]] in self.state_ranges[i[1]] and states[i[1]] in self.state_ranges[i[0]]]  # noqa: E501

        if self.n_ex == 0:
            n_ex = 1    # One swap will be carried out.
            print('Note: At most only 1 swap will be carried out, which is between neighboring replicas.')
            swappables = [i for i in swappables if np.abs(i[0] - i[1]) == 1]
        elif self.n_ex == 'N^3':
            n_ex = len(swappables) ** 3
        else:
            n_ex = self.n_ex

        self.n_swap_attempts += n_ex
        print(f"Swappable pairs: {swappables}")

        try:
            swap_list = random.choices(swappables, k=n_ex)
        except IndexError:
            # In the case that swappables is an empty list, i.e. no swappable pairs.
            swap_list = []

        return swap_list

    def get_swapping_pattern(self, swap_list, dhdl_files, states, lambda_vecs, weights):
        """
        Returns a list that represents how the replicas should be swapped. The list is always
        intiliazed with :code:`[0, 1, 2, ...]` and gets updated with swap acceptance/rejection.
        For example, if the list returned is :code:`[0, 2, 1, 3]`, it means the configurations of
        replicas 1 and 2 are swapped. If it's :code:`[2, 0, 1, 3]`, then 3 replicas (indices 0, 1, 2)
        need to swap its configuration in the next iteration.

        Parameters
        ----------
        swap_list : list or None
            A list of tuples of simulation indices to be swapped. The list could be empty, which means
            there is no any swappable pair.
        dhdl_files : list
            A list of DHDL files of ALL simulations. Note that the filename should be ordered
            with ascending simulation/replica indices, i.e. the n-th filename in the list should be
            the DHDL file of the n-th simulation.
        states : list
            A list of last sampled states of ALL simulaitons. Typically generated by :obj:`.extract_final_dhdl_info`.
        lambda_vecs : list
            A list of lambda vectors corresponding to the last sampled states of ALL simulations.
            Typically generated by :obj:`.extract_final_dhdl_info`.
        weights : list
            A list of lists of final weights of ALL simulations. Typically generated by :obj:`.extract_final_log_info`.

        Returns
        -------
        swap_pattern : list
            A list that represents how the replicas should be swapped. The indices of the list correspond to the
            simulation/replica indices, and the values represent the configuration index of the corresponding
            simulation/replica. For example, if the swapping pattern is :code:`[0, 2, 1, 3]`, it means that after
            swapping, simulations/replicas with indices 0, 1, 2, and 3 should be in configurations 0, 1, 3,
            respectively.
        """
        swap_pattern = list(range(self.n_sim))   # Can be regarded as the indices of DHDL files/configurations
        if swap_list is []:
            print('No swap is proposed because there is no swappable pair at all.')
        else:
            for i in range(len(swap_list)):
                swap = swap_list[i]
                if self.verbose is True:
                    print(f'\nA swap ({i + 1}/{len(swap_list)}) is proposed between Simulation {swap[0]} (state {states[swap[0]]}) and Simulation {swap[1]} (state {states[swap[1]]}) ...')  # noqa: E501

                # For each swap, calculate the acceptance ratio and decide whether to accept the swap.
                prob_acc = self.calc_prob_acc(swap, dhdl_files, states, lambda_vecs, weights)
                swap_bool = self.accept_or_reject(prob_acc)

                # Each DHDL file corresponds to one configuration, so if the swap is accepted,
                # we switch the order of the two corresponding DHDL file names in the DHDL files.
                # Also, the indices in swap_pattern need to be update correspondingly.
                if swap_bool is True:
                    # The assignments need to be done at the same time in just one line.
                    dhdl_files[swap[0]], dhdl_files[swap[1]] = dhdl_files[swap[1]], dhdl_files[swap[0]]
                    swap_pattern[swap[0]], swap_pattern[swap[1]] = swap_pattern[swap[1]], swap_pattern[swap[0]]
                else:
                    pass

                if self.verbose is True:
                    print(f'Swapping pattern: {swap_pattern})')
                else:
                    if i == len(swap_list) - 1:
                        print(f'\n{len(swap_list)} swaps have been proposed.')
                        print(f'Swapping pattern: {swap_pattern}')

        # Update the replica trajectories
        last_states = [i[-1] for i in self.rep_trajs]
        current_states = [last_states[swap_pattern[i]] for i in range(self.n_sim)]
        for i in range(self.n_sim):   # note that self.n_sim = len(swap_pattern)
            self.rep_trajs[i].append(current_states[i])

        return swap_pattern

    def calc_prob_acc(self, swap, dhdl_files, states, lambda_vecs, weights):
        """
        Calculates the acceptance ratio given the Monte Carlo scheme for swapping the simulations.

        Parameters
        ----------
        swap : tuple
            A tuple of indices corresponding to the simulations to be swapped.
        dhdl_files : list
            A list of DHDL files of ALL simulations. Note that the filename should be ordered
            with ascending simulation/replica indices, i.e. the n-th filename in the list should be
            the DHDL file of the n-th simulation.
        states : list
            A list of last sampled states of ALL simulaitons. Typically generated by :obj:`.extract_final_dhdl_info`.
        lambda_vecs : list
            A list of lambda vectors corresponding to the last sampled states of ALL simulations.
            Typically generated by :obj:`.extract_final_dhdl_info`.
        weights : list
            A list of lists of final weights of ALL simulations. Typiecally generated by
            :obj:`.extract_final_log_info`.

        Returns
        -------
        prob_acc : float
            The acceptance ratio.
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
            self.n_rejected += 1
            if self.verbose is True:
                print("  Swap rejected! ", end="", flush=True)
        else:
            rand = random.random()
            if self.verbose is True:
                print(
                    f"  Acceptance rate: {prob_acc:.3f} / Random number drawn: {rand:.3f}"
                )
            if rand < prob_acc:
                swap_bool = True
                # Below we flush the buffer so the next STDOUT ("Swapping pattern: ...") will be appended
                if self.verbose is True:
                    print("  Swap accepted! ", end="", flush=True)
            else:
                swap_bool = False
                self.n_rejected += 1
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
        if self.N_cutoff == -1:
            print('\nNote: No histogram correction will be performed.')
        else:
            if self.verbose is True:
                print("\nPerforming histogram correction for the lambda weights ...")
            else:
                print("\nPerforming histogram correction for the lambda weights ...", end="")

            for i in range(len(weights)):  # loop over the replicas
                if self.verbose is True:
                    print(f"  Counts of rep {i}:\t\t{counts[i]}")
                    print(f'  Original weights of rep {i}:\t{[float(f"{k:.3f}") for k in weights[i]]}')

                for j in range(1, len(weights[i])):  # loop over the alchemical states
                    if counts[i][j - 1] != 0 and counts[i][j - 1] != 0:
                        if np.min([counts[i][j - 1], counts[i][j]]) > self.N_cutoff:
                            weights[i][j] += np.log(counts[i][j - 1] / counts[i][j])

                if self.verbose is True:
                    print(f'  Corrected weights of rep {i}:\t{[float(f"{k:.3f}") for k in weights[i]]}\n')

            if self.verbose is False:
                print(' Done')

        return weights

    def combine_weights(self, weights, method):
        """
        Combine alchemical weights across multiple replicas using probability ratios
        or weight differences. (See :ref:`doc_w_schemes` for mor details.)

        Parameters
        ----------
        weights : list
            A list of Wang-Landau weights of ALL simulations
        method : str
            Method for combining weights. Must be one of the following:

              * :code:`None`: No weight combination is performed, and the original weights are returned.
              * :code:`mean`: The arithmetic mean of the probability ratios is used to scale the weights.
              * :code:`geo-mean` or :code:`geo_mean`: The geometric mean of the probability ratios is
                used to scale the weights.
              * :code:`g-diff` or :code:`g_diff`: The difference between neighboring weights is used
                to determine the alchemical weights.

        Returns
        -------
        weights : list
            A list of original (if method is :code:`None`) or modified Wang-Landau weights of ALL simulations.
        g_vec : np.array
            An array of alchemical weights of the whole range of states.
        """
        if method is None:
            print('Note: No weight combination will be performed.')
            g_vec = None
            return weights, g_vec
        else:
            if self.verbose is True:
                print(f'Performing weight combination with the {method} method ...')
            else:
                print(f'Performing weight combination with the {method} method ...', end='')

            if self.verbose is True:
                w = np.round(weights, decimals=3).tolist()  # just for printing
                print('  Original weights:')
                for i in range(len(w)):
                    print(f'    Rep {i}: {w[i]}')

            if method == 'g-diff' or method == 'g_diff':
                # Method based on weight differences
                dg_vec = []
                dg_adjacent = [list(np.diff(weights[i])) for i in range(len(weights))]
                for i in range(self.n_tot - 1):
                    dg_list = []
                    for j in range(len(self.state_ranges)):
                        if i in self.state_ranges[j] and i + 1 in self.state_ranges[j]:
                            idx = self.state_ranges[j].index(i)
                            dg_list.append(dg_adjacent[j][idx])
                    dg_vec.append(np.mean(dg_list))
                dg_vec.insert(0, 0)
                g_vec = np.array([sum(dg_vec[:(i + 1)]) for i in range(len(dg_vec))])
            else:
                # Method based on probability ratios
                # Step 1: Convert the weights into probabilities
                weights = np.array(weights)
                prob = np.array([[np.exp(-i)/np.sum(np.exp(-weights[j])) for i in weights[j]] for j in range(len(weights))])  # noqa: E501

                # Step 2: Caclulate the probability ratios (after figuring out overlapped states between adjacent replicas)  # noqa: E501
                overlapped = [set(self.state_ranges[i]).intersection(set(self.state_ranges[i + 1])) for i in range(len(self.state_ranges) - 1)]  # noqa: E501
                prob_ratio = [prob[i + 1][: len(overlapped[i])] / prob[i][-len(overlapped[i]):] for i in range(len(overlapped))]  # noqa: E501

                # Step 3: Average the probability ratios
                avg_ratio = [1]   # This allows easier scaling since the first prob vector stays the same.
                if method == 'mean':
                    avg_ratio.extend([np.mean(prob_ratio[i]) for i in range(len(prob_ratio))])
                elif method == 'geo-mean':
                    avg_ratio.extend([np.prod(prob_ratio[i])**(1/len(prob_ratio[i])) for i in range(len(prob_ratio))])

                # Step 4: Scale the probabilities for each replica
                scaled_prob = np.array([prob[i] / np.prod(avg_ratio[: i + 1]) for i in range(len(prob))])

                # Step 5: Average and convert the probabilities
                p_vec = []
                for i in range(self.n_tot):   # global state index
                    p = []   # a list of probabilities to be averaged for each state
                    for j in range(len(self.state_ranges)):   # j can be regared as the replica index
                        if i in self.state_ranges[j]:
                            local_idx = i - j * self.s
                            p.append(scaled_prob[j][local_idx])
                    if method == 'mean':
                        p_vec.append(np.mean(p))
                    elif method == 'geo-mean' or method == 'geo_mean':
                        p_vec.append(np.prod(p) ** (1 / len(p)))

                g_vec = -np.log(p_vec)
                g_vec -= g_vec[0]

            # Determine the vector of alchemical weights for each replica
            weights = [list(g_vec[i: i + self.n_sub] - g_vec[i: i + self.n_sub][0]) for i in range(self.n_sim)]
            weights = np.round(weights, decimals=5).tolist()

            if self.verbose is True:
                w = np.round(weights, decimals=3).tolist()  # just for printing
                print('\n  Modified weights:')
                for i in range(len(w)):
                    print(f'    Rep {i}: {w[i]}')

            if self.verbose is False:
                print(' DONE')
                print(f'The alchemical weights of all states: \n  {list(np.round(g_vec, decimals=3))}')
            else:
                print(f'\n  The alchemical weights of all states: \n  {list(np.round(g_vec, decimals=3))}')

            return weights, g_vec

    def run_EEXE(self, n):
        """
        Makes TPR files and runs an ensemble of expanded ensemble simulations
        using GROMACS.

        Parameters
        ----------
        n : int
            The iteration index (starting from 0).

        Returns
        -------
        md : :code:`gmxapi.commandline.CommandlineOperation` obj
            The :code:`gmxapi.commandline.CommandlineOperation` object returned by :code:`gmxapi.mdrun`
            which contains STDOUT and STDERR of the simulation.

        Notes
        -----
        This function performs the following steps:

          1. Prepares TPR files for the simulation ensemble using :code:`grompp`.
          2. Runs all the simulations simultaneously using :code:`mdrun`.
          3. Removes any empty directories created by the function.

        The function assumes that the GROMACS executable is available.
        """
        if rank == 0:
            iter_str = f'\nIteration {n}: {self.dt * self.nst_sim * n: .1f} - {self.dt * self.nst_sim * (n + 1): .1f} ps'  # noqa: E501
            print(iter_str + '\n' + '=' * (len(iter_str) - 1))

        if rank == 0:
            dir_before = [
                i for i in os.listdir(".") if os.path.isdir(os.path.join(".", i))]
            print("Preparing the tpr files for the simulation ensemble ...", end="")

        grompp = gmx.commandline_operation(
            "gmx",
            arguments=["grompp"],  # noqa: E127
            input_files=[  # noqa: E127
                {
                    "-f": f"../sim_{i}/iteration_{n}/{self.mdp.split('/')[-1]}",
                    "-c": f"../sim_{i}/iteration_{n}/{self.gro.split('/')[-1]}",
                    "-p": f"../sim_{i}/iteration_{n}/{self.top.split('/')[-1]}",
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
        if rank == 0:
            print("Running an ensemble of simulations ...", end="")

        if self.parallel is True:
            tpr = [f'{grompp.output.file["-o"].result()[i]}' for i in range(self.n_sim)]
            inputs = gmx.read_tpr(tpr)
            md = gmx.mdrun(inputs, runtime_args=self.runtime_args)
            md.run()
        else:
            # Note that we could use output_files argument to customize the output file
            # names but here we'll just use the defaults.

            arguments = ['mdrun']  # arguments for gmx.commandline_operation

            if self.runtime_args is not None:
                # Turn the dictionary into a list with the keys alternating with values
                args_keys = list(self.runtime_args.keys())
                args_vals = list(self.runtime_args.values())

                add_args = []
                for i in range(len(args_keys)):
                    add_args.append(args_keys[i])
                    add_args.append(args_vals[i])

                arguments.extend(add_args)

            md = gmx.commandline_operation(
                "gmx",
                arguments=arguments,
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
