.. _doc_cli:

1. Command-line interface (CLI)
===============================
:code:`ensemble_md` provides three command-line interfaces (CLI), including :code:`explore_REXEE`, :code:`run_REXEE` and :code:`analyze_REXEE`.
:code:`explore_REXEE` helps the user to figure out possible combinations of REXEE parameters, while :code:`run_REXEE` and :code:`analyze_REXEE`
can be used to perform and analyze REXEE simulations, respectively. Below we provide more details about each of these CLIs.

.. _doc_explore_REXEE:

1.1. CLI :code:`explore_REXEE`
------------------------------
Here is the help message of :code:`explore_REXEE`:

::

    usage: explore_REXEE [-h] -N N [-r R] [-n N] [-s S] [-c] [-e]

    This CLI explores the parameter space of a homogenous REXEE simulation to help you
    figure out all possible combinations of the number of replicas, the number of states in
    each replica, and the number of overlapping states, given the total number states.

    optional arguments:
      -h, --help      show this help message and exit
      -N N, --N N     The total number of states of the REXEE simulation.
      -r R, --r R     The number of replicas that compose the REXEE simulation.
      -n N, --n N     The number of states per each replica.
      -s S, --s S     The state shift between adjacent replicas.
      -c, --cnst      Whether to apply the constraint such that the number of overlapping
                      states does not exceed 50% of the number of states in adjacent
                      replicas. (Default: False)
      -e, --estimate  Whether to provide estimates of the chance of not having any swappable
                      pairs for each solution. (Default: False)


1.2. CLI :code:`run_REXEE`
--------------------------
Here is the help message of :code:`run_REXEE`:

::

    usage: run_REXEE [-h] [-y YAML] [-c CKPT] [-g G_VECS] [-o OUTPUT] [-m MAXWARN]

    This CLI runs a REXEE simulation given necessary inputs.

    optional arguments:
      -h, --help            show this help message and exit
      -y YAML, --yaml YAML  The file path of the input YAML file that contains REXEE
                              parameters. (Default: params.yaml)
      -c CKPT, --ckpt CKPT  The file path of the NPY file containing the replica-space
                              trajectories. This file is a necessary checkpoint file for
                              extending the simulaiton. (Default: rep_trajs.npy)
      -g G_VECS, --g_vecs G_VECS
                              The file path of the NPY file containing the timeseries of the
                              whole-range alchemical weights. This file is a necessary input
                              if ones wants to update the file when extending a weight-
                              updating simulation. (Default: g_vecs.npy)
      -o OUTPUT, --output OUTPUT
                              The file path of the output file for logging how replicas
                              interact with each other. (Default: run_REXEE_log.txt)
      -m MAXWARN, --maxwarn MAXWARN
                              The maximum number of warnings in parameter specification to be
                              ignored. (Default: 0)
  
In our current implementation, it is assumed that all replicas of a REXEE simulations are performed in
parallel using MPI. Naturally, performing a REXEE simulation using :code:`run_REXEE` requires a command-line interface
to launch MPI processes, such as :code:`mpirun` or :code:`mpiexec`. For example, on a 128-core node
in a cluster, one may use :code:`mpirun -np 4 run_REXEE` (or :code:`mpiexec -n 4 run_REXEE`) to run a REXEE simulation composed of 4
replicas with 4 MPI processes. Note that in this case, it is often recommended to explicitly specify
more details about resources allocated for each replica. For example, one can specify :code:`{'-nt': 32}`
for the REXEE parameter :code:`runtime_args` in the input YAML file (see :ref:`doc_REXEE_parameters`),
so each of the 4 replicas will use 32 threads (assuming thread-MPI GROMACS), taking the full advantage
of 128 cores.

1.3. CLI :code:`analyze_REXEE`
------------------------------
Finally, here is the help message of :code:`analyze_REXEE`:

::

    usage: analyze_REXEE [-h] [-y YAML] [-o OUTPUT] [-rt REP_TRAJS] [-st STATE_TRAJS]
                        [-sts STATE_TRAJS_FOR_SIM] [-d DIR] [-m MAXWARN]

    This CLI analyzes a REXEE simulation.

    optional arguments:
      -h, --help            show this help message and exit
      -y YAML, --yaml YAML  The file path of the input YAML file used to run the REXEE
                              simulation. (Default: params.yaml)
      -o OUTPUT, --output OUTPUT
                              The file path of the output log file that contains the analysis
                              results of REXEE. (Default: analyze_REXEE_log.txt)
      -rt REP_TRAJS, --rep_trajs REP_TRAJS
                              The file path of the NPY file containing the replica-space
                              trajectory. (Default: rep_trajs.npy)
      -st STATE_TRAJS, --state_trajs STATE_TRAJS
                              The file path of the NPY file containing the stitched state-
                              space trajectory. If the specified file is not found, the code
                              will try to find all the trajectories and stitch them. (Default:
                              state_trajs.npy)
      -sts STATE_TRAJS_FOR_SIM, --state_trajs_for_sim STATE_TRAJS_FOR_SIM
                              The file path of the NPY file containing the stitched state-
                              space time series for different state sets. If the specified
                              file is not found, the code will try to find all the time series
                              and stitch them. (Default: state_trajs.npy)
      -d DIR, --dir DIR     The path of the folder for storing the analysis results.
                              (Default: analysis)
      -m MAXWARN, --maxwarn MAXWARN
                              The maximum number of warnings in parameter specification to be
                              ignored. (Default: 0)

2. Implemented workflow
=======================
In this section, we introduce the workflow implemented in the CLI :code:`run_REXEE` that can be used to 
launch REXEE simulations. While this workflow is made as flexible as possible, interested users
can use functions defined :class:`ReplicaExchangeEE` to develop their own workflow, or consider contributing
to the source code of the CLI :code:`run_REXEE`. As an example, a hands-on tutorial that uses the CLI :code:`run_REXEE` can be found in 
`Tutorial 1: Launching a REXEE simulation`_. 

.. _`Tutorial 1: Launching a REXEE simulation`: examples/run_REXEE.ipynb


Step 1: Set up parameters
-------------------------
To run a REXEE simulation in GROMACS using the CLI :code:`run_REXEE`, one at 
least needs to following four files. (Check :ref:`doc_input_files` for more details.)

* One YAML file that specifies REXEE parameters, as specified via the CLI :code:`run_REXEE`.
* One GRO file of the system of interest, as specified in the input YAML file.
* One TOP file of the system of interest, as specified in the input YAML file.
* One MDP template for customizing MDP files for different replicas, as specified in the input YAML file.

Note that multiple GRO/TOP files can be provided to initiate different replicas with different configurations/topologies,
in which case the number of GRO/TOP files must be equal to the number of replicas.
Also, the MDP template should contain parameters shared by all replicas and define the coupling parameters for all
intermediate states. Moreover, additional care needs to be taken for specifying some MDP parameters need additional care to be taken, which we describe in
:ref:`doc_mdp_params`. Lastly, to extend a REXEE simulation, one needs to additionally provide the following
two files (generated by the existing simulation) as necessary checkpoints:

* One NPY file containing the replica-space trajectories of different configurations, as specified in the input YAML file.
* One NPY file containing the timeseries of the whole-range alchemical weights, as specified in the input YAML file. This is only needed for extending a weight-updating REXEE simulation.

In the CLI :code:`run_REXEE`, the class :class:`.ReplicaExchangeEE` is instantiated with the given YAML file, where
the user needs to specify how the replicas should be set up or interact with each 
other during the simulation ensemble. Check :ref:`doc_parameters` for more details.

Step 2: Run the 1st iteration
-----------------------------
After setting things up in the previous step, the CLI :code:`run_REXEE` uses the function :obj:`.run_REXEE` to run subprocess calls to
launch GROMACS :code:`grompp` and :code:`mdrun` commands in parallel for the first iteration. 

Step 3: Set up the new iteration
--------------------------------
In the CLI :code:`run_REXEE`, this step can be further divided into the following substeps.

Step 3-1: Extract the final status of the previous iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To calculate the acceptance ratio and modify the mdp files in later steps, we first need to extract the information
of the final status of the previous iteration. Specifically, for all the replica simulations, we need to

* Find the last sampled state and the corresponding lambda values from the DHDL files, which are necessary for both fixed-weight and weight-updating simulations.
* Find the final Wang-Landau incrementors and weights from the LOG files, which are necessary for a weight-updating simulation.

These two tasks are done by :obj:`.extract_final_dhdl_info` and :obj:`.extract_final_log_info`.

.. _doc_swap_basics:

Step 3-2: Identify the swapping pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given the information of the final status of the previous simulation, the CLI :code:`run_REXEE` runs the function :obj:`.get_swapping_pattern` to figure out how the coordinates should be swapped between replicas.
Specifically, the function does the following:

- Identify swappable pairs using the function :obj:`.identify_swappable_pairs`. Notably, replicas can be
  swapped only if the states to be swapped are present in both of the state sets
  corresponding to the two replicas. This definition automatically implies one necessary but not sufficient condition that 
  the replicas to be swapped should have overlapping state sets. Practically, if the states to be swapped are 
  not present in both state sets, potential energy differences required for the calculation of :math:`\Delta`
  will not be available, which makes the calculation of the acceptance ratio technically impossible.
- Propose a swap using the function :obj:`.propose_swap`.
- Calculates the acceptance ratio using :math:`\Delta u` values
  obtained from the DHDL files using the function :obj:`.calc_prob_acc`.
- Use the funciton :obj:`.accept_or_reject` to draw a random number and compare with the acceptance ratio
  to decide whether the swap should be accepted or not. 
- Propose and evaluate multiple swaps if needed (e.g., when the exhaustive exchange proposal scheme is used), and finally returns a list
  that represents how the configurations should be swapped in the next iteration. 

For more details, please refer to the API documentation of the involved functions.

Step 3-3: Apply correction schemes if needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For a weight-updating REXEE simulation, correction schemes may be applied if specified. Specifically,
the CLI :code:`run_REXEE` applies the weight combination scheme using the function :obj:`.combine_weights`
and the histogram correction scheme using the function :obj:`.histogram_correction`.
For more details about correction schemes, please refer to the section :ref:`doc_correction`.

Step 3-4: Set up the input files for the next iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the final configuration has been figured out by :obj:`.get_swapping_pattern` (and the weights/counts have been adjusted by the specified correction schemes, if any),
the CLI :code:`run_REXEE` sets up input files for the next iteration. In principle, the new iteration should inherit the final
status of the previous iteration. 
This means:

* For each replica, the input configuration for initializing a new iteration should be the output configuration of the previous iteration. For example,
  if the final configurations are represented by :code:`[1, 2, 0, 3]` (returned by :obj:`.get_swapping_pattern`), then in the next iteration, replica 0
  should be initialized by the output configuration of replica 1 in the previous iteration, while replica 3 can just inherit the output configuration from
  previous iteration of the same replica. Notably, instead of exchanging the MDP files, the CLI :code:`run_REXEE` swaps out the coordinate files to exchange
  replicas, which is equivalent to exchanging the MDP files.
* For each replica, the MDP file for the new iteration should be the same as the one used in the previous iteration of the same replica except that parameters
  like :code:`tinit`, :code:`init_lambda_state`, :code:`init_wl_delta`, and :code:`init_lambda_weights` should be modified to the final values in the previous
  iteration. In the CLI :code:`run_REXEE`, this is done by :obj:`.update_MDP`.

Step 4: Run the new iteration
-----------------------------
After the input files for a new iteration have been set up, we use the procedure in Step 2 to 
run a new iteration. Then, the CLI :code:`run_REXEE` loops between Steps 3 and 4 until the desired number of 
iterations (:code:`n_iterations` specified in the input YAML file) is reached. 

.. _doc_parameters:

3. Input YAML parameters
========================
In the current implementation of the algorithm, 30 parameters can be specified in the input YAML file.
Note that the two CLIs :code:`run_REXEE` and :code:`analyze_REXEE` share the same input YAML file, so we also
include parameters for data analysis here.

3.1. GROMACS executable
-----------------------

  - :code:`gmx_executable`: (Optional, Default: :code:`'gmx_mpi'`)
      The GROMACS executable to be used to run the REXEE simulation. The value could be as simple as :code:`gmx`
      or :code:`gmx_mpi` if the exeutable has been sourced. Otherwise, the full path of the executable (e.g.,
      :code:`/usr/local/gromacs/bin/gmx`, the path returned by the command :code:`which gmx`) should be used.
      Currently, our implementation only works with thread-MPI GROMACS. Implementation that works with MPI-enabled
      GROMACS will be released soon. (Check `Issue 20`_ for the current progress.)

.. _`Issue 20`: https://github.com/wehs7661/ensemble_md/issues/20


.. _doc_input_files:

3.2. Input files
----------------

  - :code:`gro`: (Required)
      The path of the input system configuration in the form of GRO file(s) used to initiate the REXEE simulation. If only one GRO file is specified,
      it will be used to initiate all the replicas. If multiple GRO files are specified (using the YAML syntax),
      the number of GRO files has to be the same as the number of replicas. 
  - :code:`top`: (Required)
      The path of the input system topology in the form of TOP file(s) used to initiate the REXEE simulation. If only one TOP file is specified,
      it will be used to initiate all the replicas. If multiple TOP files are specified (using the YAML syntax),
      the number of TOP files has to be the same as the number of replicas. In the case where multiple TOP and GRO files are specified,
      the i-th TOP file corresponds to the i-th GRO file.
  - :code:`mdp`: (Required)
      The path of the input MDP file used to initiate the REXEE simulation. Specifically, this input MDP file will serve as a template for
      customizing MDP files for all replicas. Therefore, the MDP template must specify the whole range of :math:`λ` values
      and :math:`λ`-relevant parameters. This holds for REXEE simulations for multiple serial mutations as well.
      For example, in a REXEE simulation that mutates methane to ethane in one replica and ethane to propane in the other replica, if
      exchanges only occur in the end states, then one could have :math:`λ` values like :code:`0.0 0.3 0.7 1.0 0.0 0.3 ...`. Notably, unlike
      the parameters :code:`gro` and :code:`top`, only one MDP file can be specified for the parameter :code:`mdp`. If you wish to use
      different parameters for different replicas, please use the parameter :code:`mdp_args`.
  - :code:`modify_coords`: (Optional, Default: :code:`None`)
      The file path of the Python module for modifying the output coordinates of the swapping replicas
      before the coordinate exchange, which is generally required in multi-topology REXEE simulations.
      For the CLI :code:`run_REXEE` to work, here is the predefined contract for the module/function based on the assumptions :code:`run_REXEE` makes.
      Modules/functions not obeying the contract are unlikely to work.

        - Multiple functions can be defined in the module, but the function for coordinate manipulation must have the same name as the module itself.
        - The function must only have two compulsory arguments, which are the two GRO files to be modified. The function must not depend on the order of the input GRO files. 
        - The function must return :code:`None` (i.e., no return value). 
        - The function must save the modified GRO file as :code:`confout.gro`. Specifically, if :code:`directory_A/output.gro` and :code:`directory_B/output.gro` are input, then :code:`directory_A/confout.gro` and :code:`directory_B/confout.gro` must be saved. (For more information, please visit `Tutorial 3: Multi-topology REXEE (MT-REXEE) simulations`_.) Note that in the CLI :code:`run_REXEE`, :code:`confout.gro` generated by GROMACS will be automatically renamed with a :code:`_backup` suffix to prevent overwriting.

.. _`Tutorial 3: Multi-topology REXEE (MT-REXEE) simulations`: examples/run_REXEE_modify_inputs.ipynb
        
.. _doc_REXEE_parameters:

3.3. REXEE parameters
---------------------

  - :code:`n_sim`: (Required)
      The number of replica simulations.
  - :code:`n_iter`: (Required)
      The number of iterations. In a REXEE simulation, one iteration means one exchange interval, which can involve multiple proposed swaps
      (if the exhaustive exchange proposal scheme is used). Note that when extending a simulation is desired and the necessary checkpoint files are provided,
      this parameter takes into account the number of iterations that have already been performed. That is, if a simulation has already been performed for 100 iterations,
      and one wants to extend it for 50 more iterations, then the value of this parameter should be 150.
  - :code:`s`: (Required)
      The shift in the state sets between adjacent replicas. For example, if replica 1 samples states 0, 1, 2, 3 and replica 2 samples
      states, 2, 3, 4, 5, then :code:`s = 2` should be specified.
  - :code:`nst_sim`: (Optional, Default: :code:`nsteps` in the template MDP file)
      The number of simulation steps to carry out for one iteration. The value specified here will
      overwrite the :code:`nsteps` parameter in the MDP file of each iteration. Note that this option assumes replicas with homogeneous simulation lengths.
  - :code:`add_swappables`: (Optional, Default: :code:`None`)
      A list of lists that additionally consider states (in global indices) that can be swapped. For example, :code:`add_swappables=[[4, 5], [14, 15]]` means that
      if a replica samples state 4, it can be swapped with another replica that samples state 5 and vice versa. The same logic applies to states 14 and 15. 
      This could be useful for multi-topology REXEE (MT-REXEE) simulations, where we enforce the consideration of exchanges between states 4 and 5 (and 14 and 15) and perform
      coordinate manipulation when necessary.
  - :code:`proposal`: (Optional, Default: :code:`'exhaustive'`)
      The method for proposing simulations to be swapped. Available options include :code:`single`, :code:`neighboring`, and :code:`exhaustive`.
      For more details, please refer to :ref:`doc_proposal`.
  - :code:`w_combine`: (Optional, Default: :code:`False`)
      Whether to perform weight combination or not. Note that weights averaged over from the last updated of the Wang-Landau incrementor (instead of the
      final weights) will be used for weight combination. For more details about, please refer to :ref:`doc_w_schemes`.
  - :code:`w_mean_type`: (Optional, Default: :code:`'simple'`)
      The type of mean to use when combining weights. Available options include :code:`simple` and :code:`weighted`.
      For the later case, inverse-variance weighted means are used. For more details about, please refer to :ref:`doc_w_schemes`.
  - :code:`N_cutoff`: (Optional, Default: 1000)
      The histogram cutoff for weight corrections. A cutoff of 1000 means that weight corrections will be applied only if
      the counts of the involved states are both larger than 1000. A value of -1 means that no weight correction will be performed.
      For more details, please please refer to :ref:`doc_weight_correction`.
  - :code:`hist_corr` (Optional, Default: :code:`False`)
      Whether to perform histogram correction. For more details, please refer to :ref:`doc_hist_correction`.
  - :code:`mdp_args`: (Optional, Default: :code:`None`)
      A dictionary that contains MDP parameters differing across replicas. For each key in the dictionary, the value should
      always be a list of length of the number of replicas. For example, :code:`{'ref_p': [1.0, 1.01, 1.02, 1.03]}` means that the
      MDP parameter :code:`ref_p` will be set as 1.0 bar, 1.01 bar, 1.02 bar, and 1.03 bar for replicas 0, 1, 2, and 3, respectively.
      Note that while this feature allows high flexibility in parameter specification, not all parameters are suitable to be
      varied across replicas. Users should use this parameter with caution, as there is no check for the validity of the MDP parameters.
      Additionally, this feature is a work in progress and differing :code:`ref_t` or :code:`dt` across replicas would not work. 
  - :code:`grompp_args`: (Optional: Default: :code:`None`)
      A dictionary that contains additional arguments to be appended to the GROMACS :code:`grompp` command. 
      For example, one could have :code:`{'-maxwarn', '1'}` to specify the :code:`maxwarn` argument for the :code:`grompp` command.
  - :code:`runtime_args`: (Optional, Default: :code:`None`)
      A dictionary that contains additional runtime arguments to be appended to the GROMACS :code:`mdrun` command.
      For example, one could have :code:`{'-nt': 16}` to run the simulation using tMPI-enabled GROMACS with 16 threads.

3.4. Output settings
--------------------
  - :code:`verbose`: (Optional, Default: :code:`True`)
      Whether a verbse log file is desired. 
  - :code:`n_ckpt`: (Optional, Default: 100)
      The number of iterations between each checkpoint. Specifically, the CLI :code:`run_REXEE` will save the replica-space trajectories
      and the timeseries of the whole-range alchemical weights (in a weight-updating simulation) every :code:`n_ckpt` iterations. This is useful for extending a simulation.
  - :code:`rm_cpt`: (Optional, Default: :code:`True`)
      Whether the GROMACS checkpoint file (:code:`state.cpt`) from each iteration should be deleted.
      Normally we don't need GROMACS CPT files for REXEE simulations (even for extension) so we recommend just
      deleting the CPT files (which could save a lot of space if you perform a huge number of iterations).
      If you wish to keep them, specify this parameter as :code:`False`.
  
.. _doc_analysis_params:

3.5. Data analysis
------------------
  - :code:`msm`: (Optional, Default: :code:`False`)
      Whether to build Markov state models (MSMs) for the REXEE simulation and perform relevant analysis.
  - :code:`free_energy`: (Optional, Default: :code:`False`)
      Whether to perform free energy calculations or not.
  - :code:`subsampling_avg`: (Optional, Default: :code:`False`)
      Whether to take the arithmetic average of the truncation fractions and the geometric average of the
      statistical inefficiencies over replicas when subsampling data for free energy calculations. For systems
      where the sampling is challenging, the truncation fraction or statistical inefficiency may vary largely
      across state sets, in which case this option could be useful.
  - :code:`df_spacing`: (Optional, Default: 1)
      The spacing (in the number of data points) to consider when subsampling the data, which is assumed to
      be the same for all replicas.
  - :code:`df_ref`: (Optional, Default: :code:`None`)
      The reference free energy profile for the whole range of states. The input should be a list having the length of the total number of states.
  - :code:`df_method`: (Optional, Default: :code:`'MBAR'`)
      The free energy estimator to use in free energy calculations. Available options include :code:`'TI'`, :code:`'BAR'`, and :code:`'MBAR'`.
  - :code:`err_method`: (Optional, Default: :code:`'propagate'`)
      The method for estimating the uncertainty of the free energy combined across multiple replicas. 
      Available options include :code:`'propagate'` and :code:`'bootstrap'`. The boostrapping method is more accurate but much more 
      computationally expensive than simple error propagation.
  - :code:`n_bootstrap`: (Optional, Default: 50)
      The number of bootstrap iterations to perform when estimating the uncertainties of the free energy differences.
  - :code:`seed`: (Optional, Default: :code:`None`)
      The random seed to use in bootstrapping.

3.6. A template input YAML file
-------------------------------
For convenience, here is a template of the input YAML file, with each optional parameter specified with the default and required 
parameters left with a blank. Note that specifying :code:`null` is the same as leaving the parameter unspecified (i.e., :code:`None`).

.. code-block:: yaml

    # Section 1: Runtime configuration
    gmx_executable: 'gmx_mpi'

    # Section 2: Input files
    gro:
    top:
    mdp:
    modify_coords: null

    # Section 3: REXEE parameters
    n_sim:
    n_iter:
    s:
    nst_sim: null
    add_swappables: null
    proposal: 'exhaustive'
    w_combine: False
    w_mean_type: 'simple'
    N_cutoff: 1000
    hist_corr: False
    mdp_args: null
    grompp_args: null
    runtime_args: null

    # Section 4: Output settings
    verbose: True
    n_ckpt: 100
    rm_cpt: True

    # Section 5: Data analysis
    msm: False
    free_energy: False 
    subsampling_avg: False
    df_spacing: 1
    df_ref: null
    df_method: 'MBAR'
    err_method: 'propagate'
    n_bootstrap: 50
    seed : null

.. _doc_mdp_params:

4. Input MDP parameters
=======================
As mentioned above, a template MDP file should have all the parameters that will be shared
across all replicas. It should also define the coupling parameters for the whole range of
states so that different MDP files can be customized for different replicas. For a REXEE simulation
launched by the CLI :code:`run_REXEE`, any GROMACS MDP parameter that could potentially lead to issues
in the REXEE simulation will raise a warning. If the number of warnings is larger than the value
specified for the flag `-m`/`--maxwarn` in the CLI :code:`run_REXEE`, the simulation will error
out. To avoid warnings arised from MDP specification, we need to take extra care for the following
MDP parameters:

- We recommend setting :code:`lmc_seed = -1` so that a different random seed
  for Monte Carlo moves in the state space will be used for each iteration. 
- We recommend setting :code:`gen_vel = yes` to re-generating new velocities for each iteration to avoid
  potential issues with detailed balance. 
- We recommend setting :code:`gen_seed = -1` so that a different random seed for velocity generation
  will be used for each iteration.
- The MDP parameter :code:`nstlog` must be a factor of the YAML parameter :code:`nst_sim` so that the final status
  of the simulation can be correctly parsed from the LOG file.
- The MDP parameter :code:`nstdhdl` must be a factor of the YAML parameter :code:`nst_sim` so that the time series
  of the state index can be correctly parsed from the DHDL file.
- In REXEE, the MDP parameter :code:`nstdhdl` must be a factor of the MDP parameter :code:`nstexpanded`, or
  the calculation of the acceptance ratio may be wrong. 
- Be careful with the pull code specification if you want to apply a distance restraint between two pull groups.
  Specifically, in a REXEE simulation, all iterations should use the same reference distance. Otherwise, poor sampling
  can be observed in a fixed-weight REXEE simulation and the equilibration time may be much longer for a weight-updating
  REXEE simulation. To ensure the same reference distance across all iterations in a REXEE simulation, consider the
  following scenarios:

    - If you would like to use the COM distance between the pull groups in the input GRO file as the reference distance
      for all the iterations (whatever that value is), then specify :code:`pull_coord1_start = yes` with
      :code:`pull_coord1_init = 0` in your input MDP template. In this case, :obj:`.update_MDP` will parse :code:`pullx.xvg`
      from the first iteration to get the initial COM distance (:code:`d`) and use it as the reference distance for all the following
      iterations using :code:`pull_coord1_start = no` with :code:`pull_coord1_init = d`. Note that this implies that
      the MDP parameter :code:`pull_nstxout` should not be 0.
    - If you want to explicitly specify a reference distance (:code:`d`) to use for all iterations, simply use 
      :code:`pull_coord1_start = no` with :code:`pull_coord1_init = d` in your input MDP template.
