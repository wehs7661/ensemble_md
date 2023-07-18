.. _doc_cli:

1. Command-line interface (CLI)
===============================
:code:`ensemble_md` provides three command-line interfaces (CLI), including :code:`explore_EEXE`, :code:`run_EEXE` and :code:`analyze_EEXE`.
:code:`explore_EEXE` helps the user to figure out possible combinations of EEXE parameters, while :code:`run_EEXE` and :code:`analyze_EEXE`
can be used to perform and analyze EEXE simulations, respectively. Below we provide more details about each of these CLIs.

1.1. CLI :code:`explore_EEXE`
-----------------------------
Here is the help message of :code:`explore_EEXE`:

::

    usage: explore_EEXE [-h] -N N [-r R] [-n N] [-s S] [-c] [-e]

    This code explores the parameter space of homogenous EEXE to help you figure out all
    possible combinations of the number of replicas, the number of states in each replica,
    and the number of overlapping states, and the total number states.

    optional arguments:
      -h, --help      show this help message and exit
      -N N, --N N     The total number of states of the EEXE simulation.
      -r R, --r R     The number of replicas that compose the EEXE simulation.
      -n N, --n N     The number of states for each replica.
      -s S, --s S     The state shift between adjacent replicas.
      -c, --cnst      Whether the apply the constraint such that the number of overlapping
                      states does not exceed 50% of the number of states in both overlapping
                      replicas.
      -e, --estimate  Whether to provide estimates of the chance of not having any swappable
                      pairs for each solution.


1.2. CLI :code:`run_EEXE`
-------------------------
Here is the help message of :code:`run_EEXE`:

::

    usage: run_EEXE [-h] [-y YAML] [-c CKPT] [-g G_VECS] [-o OUTPUT] [-m MAXWARN]

    This code runs an ensemble of expanded ensemble given necessary inputs.

    optional arguments:
    -h, --help            show this help message and exit
    -y YAML, --yaml YAML  The input YAML file that contains EEXE parameters. (Default:
                            params.yaml)
    -c CKPT, --ckpt CKPT  The NPY file containing the replica-space trajectories. This file
                            is a necessary checkpoint file for extending the simulaiton.
                            (Default: rep_trajs.npy)
    -g G_VECS, --g_vecs G_VECS
                            The NPY file containing the timeseries of the whole-range
                            alchemical weights. This file is a necessary input if ones wants
                            to update the file when extending the simulation. (Default:
                            g_vecs.npy)
    -o OUTPUT, --output OUTPUT
                            The output file for logging how replicas interact with each
                            other. (Default: run_EEXE_log.txt)
    -m MAXWARN, --maxwarn MAXWARN
                            The maximum number of warnings in parameter specification to be
                            ignored.

In our current implementation, it is assumed that all replicas of an EEXE simulations are performed in
parallel using MPI. Naturally, performing an EEXE simulation using :code:`run_EEXE` requires a command-line interface
to launch MPI processes, such as :code:`mpirun` or :code:`mpiexec`. For example, on a 128-core node
in a cluster, one may use :code:`mpirun -np 4 run_EEXE` (or :code:`mpiexec -n 4 run_EEXE`) to run an EEXE simulation composed of 4
replicas with 4 MPI processes. Note that in this case, it is often recommended to explicitly specify
more details about resources allocated for each replica. For example, one can specifies :code:`{'-nt': 32}`
for the EEXE parameter `runtime_args` (specified in the input YAML file, see :ref:`doc_EEXE_parameters`),
so each of the 4 replicas will use 32 threads (assuming thread-MPI GROMACS), taking the full advantage
of 128 cores.

1.3. CLI :code:`analyze_EEXE`
-----------------------------
Finally, here is the help message of :code:`analyze_EEXE`:

::

    usage: analyze_EEXE [-h] [-y YAML] [-o OUTPUT] [-rt REP_TRAJS] [-st STATE_TRAJS]
                        [-d DIR] [-m MAXWARN]

    This code analyzes an ensemble of expanded ensemble. Note that the template MDP file
    specified in the YAML file needs to be available in the working directory.

    optional arguments:
    -h, --help            show this help message and exit
    -y YAML, --yaml YAML  The input YAML file used to run the EEXE simulation. (Default:
                            params.yaml)
    -o OUTPUT, --output OUTPUT
                            The output log file that contains the analysis results of EEXE.
                            (Default: analyze_EEXE_log.txt)
    -rt REP_TRAJS, --rep_trajs REP_TRAJS
                            The NPY file containing the replica-space trajectory. (Default:
                            rep_trajs.npy)
    -st STATE_TRAJS, --state_trajs STATE_TRAJS
                            The NPY file containing the stitched state-space trajectory. If
                            the specified file is not found, the code will try to find all
                            the trajectories and stitch them. (Default: state_trajs.npy)
    -d DIR, --dir DIR     The name of the folder for storing the analysis results.
    -m MAXWARN, --maxwarn MAXWARN
                            The maximum number of warnings in parameter specification to be
                            ignored.

2. Recommended workflow
=======================
In this section, we introduce the workflow adopted by the CLI :code:`run_EEXE` that can be used to 
launch EEXE simulations. While this workflow is made as flexible as possible, interested users
can use functions defined :class:`ensemble_EXE` to develop their own workflow, or consider contributing
to the source code of the CLI :code:`run_EEXE`. As an example, a hands-on tutorial that uses this workflow (using the CLI :code:`run_EEXE`) can be found in 
`Tutorial 1: Ensemble of expanded ensemble`_. 

.. _`Tutorial 1: Ensemble of expanded ensemble`: examples/EEXE_tutorial.ipynb


Step 1: Set up parameters
-------------------------
To run an ensemble of expanded ensemble in GROMACS using :code:`run_EEXE.py`, one at 
least needs to following four files:

* One GRO file of the system of interest
* One TOP file of the system of interest
* One MDP template for customizing different MDP files for different replicas. 
* One YAML file that specify the EEXE-relevant parameters.

Currently, we only allow all replicas to be initiated with the same configuration represented 
by the single GRO file, but the user should also be able to initialize different replicas with different 
configurations (represented by multiple GRO files) in the near future. Also, the MDP template should contain parameters 
common across all replicas and define the coupling parmaeters for all possible intermediate states,
so that we can cusotmize different MDP files by defining a subset of alchemical states in different 
replicas. Importantly, to extend an EEXE simulation, one needs to additionally provide the following
two checkpoint files:

* One NPY file containing the replica-space trajectories of different configurations saved by the previous run of EEXE simulation with a default name as :code:`rep_trajs.npy`.
* One NPY file containing the timeseries of the whole-range alchemical weights saved by the previous run of EEXE simulation with a default name as :code:`g_vecs.npy`.

In :code:`run_EEXE.py`, the class :class:`.EnsembleEXE` is instantiated with the given YAML file, where
the user needs to specify how the replicas should be set up or interact with each 
other during the simulation ensemble. Check :ref:`doc_parameters` for more details.

Step 2: Run the 1st iteration
-----------------------------
With all the input files/parameters set up in the previous run, one can use run the first iteration,
using :obj:`.run_EEXE`, which uses :code:`subprocess.run` to launch GROMACS :code:`grompp`
and :code:`mdrun` commands in parallel.

Step 3: Set up the new iteration
--------------------------------
In general, this step can be further divided into the following substeps.

Step 3-1: Extract the final status of the previous iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To calculate the acceptance ratio and modify the mdp files in later steps, we first need to extract the information
of the final status of the previous iteration. Specifically, for all the replica simulations, we need to

* Find the last sampled state and the corresponding lambda values from the DHDL files
* Find the final Wang-Landau incrementors and weights from the LOG files. 

These two tasks are done by :obj:`.extract_final_dhdl_info` and :obj:`.extract_final_log_info`.

.. _doc_swap_basics:

Step 3-2: Identify swappable pairs and propose simulation swap(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the information of the final status of the previous iteration is extracted, we then identify swappable pairs.
Specifically, replicas can be swapped only if the states to be swapped are present in both of the alchemical ranges 
corresponding to the two replicas. This definition automatically implies one necessary but not sufficient condition that 
the replicas to be swapped should have overlapping alchemical ranges. Practically, if the states to be swapped are 
not present in both alchemical ranges, information like :math:`\Delta U^i=U^i_n-U^j_m` will not be available 
in either DHDL files and terms like :math:`\Delta g^i=g^i_n-g^i_m` cannot be calculated from the LOG files as well, which 
makes the calculation of the acceptance ratio technicaly impossible. (For more details about the acceptance ratio is calculated
in different schemes for swapping, check the section :ref:`doc_acceptance`.) After the swappable pairs are identified, 
the user can propose swap(s) using :obj:`propose_swaps`. Swap(s) will be proposed given the specified proposal scheme (see
more details about available proposal schemes in :ref:`doc_proposal`). 

Step 3-3: Decide whether to reject/accept the swap(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step is mainly done by :obj:`.get_swapped_configs`, which calls functions :obj:`.calc_prob_acc` and :obj:`.accept_or_reject`. 
The former calculates the acceptance ratio from the DHDL/LOG files of the swapping replicas, while the latter draws a random number 
and compare with the acceptance ratio to decide whether the proposed swap should be accepted or not. If mutiple swaps are wanted,
in :obj:`.get_swapped_configs`, the acceptance ratio of each swap will be evaluated so to decide whether the swap should be accepted
or not. Based on this :obj:`get_swapped_configs` returns a list of indices that represents the final configurations after all the swaps. 

Step 3-4: Combine the weights if needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the states that are present in the alchemical ranges of multiple replicas, it is likely that they are 
sampled more frequenly overall. To leverage the fact that we collect more statistics for these states, it is recoomended 
that the weights of these states be combined across all replicas that sampled these states. This task can be completed by
:obj:`combine_wieghts`, with the desired method specified in the input YAML file. For more details about different 
methods for combining weights across different replicas, please refer to the section :ref:`doc_w_schemes`.

Step 3-5: Modify the MDP files and swap out the GRO files (if needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the final configuration has been figured out by :obj:`get_swapped_configs` (and weights have bee combined by :obj:`combine_weights`
when needed), the user should set up the input files for the next iteration. In principle, the new iteration should inherit the final
status of the previous iteration. 
This means:

* For each replica, the input configuration for initializing a new iterations should be the output configuraiton of the previous iteration. For example, if the final configurations are represented by :code:`[1, 2, 0, 3]` (returned by :obj:`.get_swapped_configs`), then in the next iteration, replica 0 should be initialized by the output configuration of replica 1 in the previous iteration, while replica 3 can just inherit the output configuration from previous iteration of the same replica. Notably, instead of exchanging the MDP files, we recommend swapping out the coordinate files to exchange replicas.
* For each replica, the MDP file for the new iteration should be the same as the one used in the previous iteartion of the same replica except that parameters like :code:`tinit`, :code:`init_lambda_state`, :code:`init_wl_delta`, and :code:`init_lambda_weights` should be modified to the final values in the previous iteration. This can be done by :class:`.gmx_parser.MDP` and :obj:`.update_MDP`.

Step 4: Run the new iteration
-----------------------------
After the input files for a new iteration have been set up, we use the procedure in Step 2 to 
run a new iteration. Then, the user should loop between Steps 3 and 4 until the desired number of 
iterations (:code:`n_iterations`) is reached. 

.. _doc_parameters:

3. Simulation parameters
========================
In the current implementation of the algorithm, 22 parameters can be specified in the input YAML file.
Note that the two CLIs :code:`run_EEXE` and :code:`analyze_EEXE` share the same input YAML file, so we also
include parameters for data analysis here.

3.1. GROMACS executable
-----------------------

  - :code:`gmx_executable`: (Optional, Default: :code:`gmx_mpi`)
      The GROMACS executable to be used to run the EEXE simulation. The value could be as simple as :code:`gmx`
      or :code:`gmx_mpi` if the exeutable has been sourced. Otherwise, the full path of the executable (e.g.
      :code:`/usr/local/gromacs/bin/gmx`, the path returned by the command :code:`which gmx`) should be used.
      Note that EEXE only works with MPI-enabled GROMACS. 

3.2. Input files
----------------

  - :code:`gro`: (Required)
      The input system configuration in the form of GRO file(s) used to initiate the EEXE simulation. If only one GRO file is specified,
      it will be used to initiate all the replicas. If multiple GRO files are specified (using the YAML syntax),
      the number of GRO files has to be the same as the number of replicas. 
  - :code:`top`: (Required)
      The input system topology in the form of TOP file(s) used to initiate the EEXE simulation. If only one TOP file is specified,
      it will be used to initiate all the replicas. If multiple TOP files are specified (using the YAML syntax),
      the number of TOP files has to be the same as the number of replicas. In the case where multiple TOP and GRO files are specified,
      the i-th TOP file corresponds to the i-th GRO file.
  - :code:`mdp`: (Required)
      The input MDP file used to initiate the EEXE simulation. Specifically, this input MDP file will serve as a template for
      customizing MDP files for all replicas. Therefore, the MDP template must have the whole range of :math:`λ` values. 
      and the corresponding weights (in fixed-weight simulations). This holds for EEXE simulations for multiple serial mutations as well.
      For example, in an EEXE simulation that mutates methane to ethane in one replica and ethane to propane in the other replica, if
      exchanges only occur in the end states, then one could have :math:`λ` values like :code:`0.0 0.3 0.7 1.0 0.0 0.3 ...`. Notably, unlike
      the parameters :code:`gro` and :code:`top`, only one MDP file can be specified for the parameter :code:`mdp`. If you wish to use
      different parameters for different replicas, please use the parameter :code:`mdp_args`.
  - :code:`modify_coords`: (Optional)
      The name of the Python module (without including the :code:`.py` extension) for modifying the output coordinates of the swapping replicas
      before the coordinate exchange, which is generally required in EEXE simulations for multiple serial mutations.
      For the CLI :code:`run_EEXE` to work, here is the predefined contract for the module/function based on the assumptions :code:`run_EEXE` makes.
      Modules/functions not obeying the contract are unlikely to work.

        - Multiple functions can be defined in the module, but the function for coordinate manipulation must have the same name as the module itself.
        - The function must only have two compulsory arguments, which are the two GRO files to be modified. The function must not depend on
            the order of the input GRO files. 
        - The function must return :code:`None` (i.e., no return value). 
        - The function is expected to save the modified GRO file as :code:`confout.gro`. Note that in the CLI :code:`run_EEXE`,
            :code:`confout.gro` generated as the simulation output will be automatically backed up to prevent overwriting.
        
.. _doc_EEXE_parameters:

3.3. EEXE parameters
--------------------

  - :code:`n_sim`: (Required)
      The number of replica simulations.
  - :code:`n_iter`: (Required)
      The number of iterations. In an EEXE simulation, one iteration means one exchange attempt. Notably, this can be used to extend the EEXE simulation.
      For example, if one finishes an EEXE simulation with 10 iterations (with :code:`n_iter=10`) and wants to continue the simulation from iteration 11 to 30,
      setting :code:`n_iter` in the next execution of :code:`run_EEXE` should suffice.
  - :code:`s`: (Required)
      The shift in the alchemical ranges between adjacent replicas (e.g. :math:`s = 2` if :math:`λ_2 = (2, 3, 4)` and :math:`λ_3 = (4, 5, 6)`.
  - :code:`nst_sim`: (Optional, Default: :code:`nsteps` in the template MDP file)
      The number of simulation steps to carry out for one iteration, i.e. stpes between exchanges proposed between replicas. The value specified here will
      overwrite the :code:`nsteps` parameter in the MDP file of each iteration. This option also assumes replicas with homogeneous simulation lengths.
  - :code:`add_swappables`: (Optional, Default: :code:`None`)
      A list of lists that additionally consider states (in global indices) that can be swapped. For example, :code:`add_swappables=[[4, 5], [14, 15]]` means that
      if a replica samples state 4, it can be swapped with another replica that samples state 5 and vice versa. The same logic applies to states 14 and 15. 
      This could be useful for EEXE simulations for multiple serial mutations, where we enforce exchanges between states 4 and 5 (and 14 and 15) and perform
      coordinate manipulation.
  - :code:`proposal`: (Optional, Default: :code:`exhaustive`)
      The method for proposing simulations to be swapped. Available options include :code:`single`, :code:`exhaustive`, :code:`neighboring`, and :code:`multiple`.
      For more details, please refer to :ref:`doc_proposal`.
  - :code:`acceptance`: (Optional, Default: :code:`metropolis`)
      The Monte Carlo method for swapping simulations. Available options include :code:`same-state`/:code:`same_state`, :code:`metropolis`, and :code:`metropolis-eq`/:code:`metropolis_eq`. 
      For more details, please refer to :ref:`doc_acceptance`.
  - :code:`w_combine`: (Optional, Default: :code:`False`)
      Whether to combine weights across multiple replicas for an weight-updating EEXE simulations. 
      For more details, please refer to :ref:`doc_w_schemes`.
  - :code:`N_cutoff`: (Optional, Default: 1000)
      The histogram cutoff. -1 means that no histogram correction will be performed.
  - :code:`n_ex`: (Optional, Default: 1)
      The number of attempts swap during an exchange interval. This option is only relevant if the option :code:`proposal` is :code:`multiple`.
      Otherwise, this option is ignored. For more details, please refer to :ref:`doc_multiple_swaps`.
  - :code:`mdp_args`: (Optional, Default: :code:`None`)
      MDP parameters differing across replicas provided in a dictionary. For each key in the dictionary, the value should
      always be a list of length of the number of replicas. For example, :code:`{'ref_p': [1.0, 1.01, 1.02, 1.03]}` means that the
      MDP parameter :code:`ref_p` will be set as 1.0 bar, 1.01 bar, 1.02 bar, and 1.03 bar for replicas 0, 1, 2, and 3, respectively.
      Note that while this feature allows high flexibility in parameter specification, not all parameters are suitable to be
      varied across replicas. For example, varying :code:`nsteps` across replicas for synchronous EEXE simulations does not make sense. 
      Additionally, this feature is a work in progress and differing :code:`ref_t` or :code:`dt` across replicas might cause issues. 
  - :code:`grompp_args`: (Optional: Default: :code:`None`)
      Additional arguments to be appended to the GROMACS :code:`grompp` command provided in a dictionary.
      For example, one could have :code:`{'-maxwarn', '1'}` to specify the :code:`maxwarn` argument for the :code:`grompp` command.
  - :code:`runtime_args`: (Optional, Default: :code:`{}`)
      Additional runtime arguments to be appended to the GROMACS :code:`mdrun` command provided in a dictionary. 
      For example, one could have :code:`{'-nt': 16}` to run the simulation using tMPI-enabled GROMACS with 16 threads.
      Notably, if MPI-enabled GROMACS is used, one should specify :code:`-np` to better use the resources. If it is
      not specified, the default will be the number of simulations and a warning will occur.

3.4. Output settings
--------------------
  - :code:`verbose`: (Optional, Default: :code:`True`)
      Whether a verbse log is wanted. 
  - :code:`n_ckpt`: (Optional, Default: 100)
      The frequency for checkpointing in the number of iterations.
  
.. _doc_analysis_params:

3.5. Data analysis
------------------
  - :code:`msm`: (Optional, Default: :code:`False`)
      Whether to build Markov state models (MSMs) for the EEXE simulation and perform relevant analysis.
  - :code:`free_energy`: (Optional, Default: :code:`False`)
      Whether to perform free energy calculations in data analysis or not. Note that free energy calculations 
      could be computationally expensive depending on the relevant settings.
  - :code:`df_spacing`: (Optional, Default: 1)
      The step to used in subsampling the DHDL data in free energy calculations.
  - :code:`df_method`: (Optional, Default: :code:`MBAR`)
      The free energy estimator to use in free energy calcuulation. Available options include :code:`TI`, :code:`BAR`, and :code:`MBAR`.
  - :code:`err_method`: (Optional, Default: :code:`propagate`)
      The method for estimating the uncertainty of the free energy combined across multiple replicas. 
      Available options include :code:`propagate` and :code:`bootstrap`. The boostrapping method is more accurate but much more 
      computationally expensive than simple error propagation.
  - :code:`n_bootstrap`: (Optional, Default: 50)
      The number of bootstrap iterations to perform when estimating the uncertainties of the free energy differences between 
      overlapping states.
  - :code:`seed`: (Optional, Default: None)
      The random seed to use in bootstrapping.

3.6. A template input YAML file
-------------------------------
For convenience, here is a template of the input YAML file, with each optional parameter specified with the default and required 
parameters left with a blank. Note that specifying :code:`null` is the same as leaving the parameter unspecified (i.e. :code:`None`).


.. code-block:: yaml

    # Section 1: Runtime configuration
    gmx_executable:

    # Section 2: Input files
    gro:
    top:
    mdp:
    modify_coords: null

    # Section 3: EEXE parameters
    n_sim:
    n_iter:
    s:
    nst_sim: null
    proposal: 'exhaustive'
    acceptance: 'metropolis' 
    w_combine: False
    N_cutoff: 1000
    n_ex: 1
    mdp_args: null
    grompp_args: null
    runtime_args: null

    # Section 4: Output settings
    verbose: True
    n_ckpt: 100

    # Section 5: Data analysis
    msm: False
    free_energy: False 
    df_spacing: 1
    df_method: "MBAR"
    err_method: "propagate"
    n_bootstrap: 50
    seed : null

