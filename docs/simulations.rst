.. _doc_cli:

Command-line interface (CLI)
============================
:code:`ensemble_md` provides three command-line interfaces (CLI), including :code:`explore_EEXE`, :code:`run_EEXE` and :code:`analyze_EEXE`.
:code:`explore_EEXE` helps the user to figure out possible combinations of EEXE parameters, while :code:`run_EEXE` and :code:`analyze_EEXE`
can be used to perform and analyze EEXE simulations, respectively. Here is the help message of :code:`explore_EEXE`:

::

    usage: explore_EEXE [-h] -N N [-r R] [-n N] [-s S] [-c]

    This code explores the parameter space of homogenous EEXE to help you figure out all
    possible combinations of the number of replicas, the number of states in each replica,
    and the number of overlapping states, and the total number states.

    optional arguments:
    -h, --help   show this help message and exit
    -N N, --N N  The total number of states of the EEXE simulation.
    -r R, --r R  The number of replicas that compose the EEXE simulation.
    -n N, --n N  The number of states for each replica.
    -s S, --s S  The state shift between adjacent replicas.
    -c, --cnst   Whether the apply the constraint such that the number of overlapping
                states does notexceed 50% of the number of states in both overlapping
                replicas.


And here is the help message of :code:`run_EEXE`:

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

Recommended workflow
====================
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
With all the input files/parameters set up in the previous run, one can use :obj:`.run_EEXE` to run the 
first iteration. Specifically, :obj:`.run_EEXE` uses :code:`gmxapi.commandline_operation` to launch an GROMACS
:code:`grompp` command to generate the input MDP file. Then, if :code:`parallel` is specified as :code:`True` 
in the input YAML file, :code:`gmxapi.mdrun` will be used to run GROMACS :code:`mdrun` commands in parallel, 
otherwise :code:`gmxapi.commandline_operation` will be used to run simulations serially.

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
in different schemes for swapping, check the section :ref:`doc_mc_schemes`.) After the swappable pairs are identified, 
the user can propose swap(s) using :obj:`propose_swaps`. Note that having multiple swaps proposed in one attempt is possible 
with :code:`n_ex` specified larger than 1 in the YAML file. In that case, :code:`n_ex` swaps will be drawn from the list of 
swappable pairs with replacement, so there is no upper limit for :code:`n_ex` and a recommended value is :math:`N^3`, where
:math:`N` is the number of replicas. If :code:`n_ex` is unspecified or specified as 0, then only 1 swap will be proposed and 
it will be between a pair of adjacent simulations (i.e. neighboring swapping). 

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
* For each replica, the MDP file for the new iteration should be the same as the one used in the previous iteartion of the same replica except that parameters like :code:`tinit`, :code:`init-lambda-state`, :code:`init-wl-delta`, and :code:`init-lambda-weights` should be modified to the final values in the previous iteration. This can be done by :class:`.gmx_parser.MDP` and :obj:`.update_MDP`.

Step 4: Run the new iteration
-----------------------------
After the input files for a new iteration have been set up, we use the procedure in Step 2 to 
run a new iteration. Then, the user should loop between Steps 3 and 4 until the desired number of 
iterations (:code:`n_iterations`) is reached. 

.. _doc_parameters:

Simulation parameters
=====================
In the current implementation of the algorithm, 22 parameters can be specified in the input YAML file.
Note that the two CLIs :code:`run_EEXE` and :code:`analyze_EEXE` share the same input YAML file, so we also
include parameters for data analysis here.

Simulation inputs
-----------------

  - :code:`gro`: (Required)
      The GRO file that contains the starting configuration for all replicas.
  - :code:`top`: (Required)
      The TOP file that contains the system topology. 
  - :code:`mdp`: (Required)
      The MDP template that has the whole range of :math:`λ` values.

EEXE parameters
---------------

  - :code:`parallel`: (Required)
      Whether the replicas of EEXE should be run in parallel or not.
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
  - :code:`mc_scheme`: (Optional, Default: :code:`metropolis`)
      The method for swapping simulations. Available options include :code:`same-state`/:code:`same_state`, :code:`metropolis`, and :code:`metropolis-eq`/:code:`metropolis_eq`. 
      For more details, please refer to :ref:`doc_mc_schemes`.
  - :code:`w_scheme`: (Optional, Default: :code:`None`)
      The method for combining weights. Available options include :code:`None` (unspecified), :code:`mean`, and :code:`geo-mean`/:code:`geo_mean`. 
      For more details, please refer to :ref:`doc_w_schemes`.
  - :code:`N_cutoff`: (Optional, Default: 1000)
      The histogram cutoff. -1 means that no histogram correction will be performed.
  - :code:`n_ex`: (Optional, Default: 0)
      The number of swaps to be proposed in one attempt. This works basically the same as :code:`-nex` flag in GROMACS replica exchange simulations. 
      A recommended value is :math:`N^3`, where :math:`N` is the number of swappable pairs and can therefore be different in each iteration. If :code:`n_ex` 
      is specified as 0, neighboring swapping will be carried out. For more details, please refer to :ref:`doc_swap_basics`.
  - :code:`runtime_args`: (Optional, Default: :code:`None`)
      Additional runtime arguments to be appended to the GROMACS :code:`mdrun` command provided in a dictionary. 
      For example, one could have :code:`{'-nt': 16}` to run the simulation using 16 threads.

Output settings
---------------
  - :code:`verbose`: (Optional, Default: :code:`True`)
      Whether a verbse log is wanted. 
  - :code:`n_ckpt`: (Optional, Default: 100)
      The frequency for checkpointing in the number of iterations.
  
.. _doc_analysis_params:

Data analysis
-------------
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

For convenience, here is a template of the input YAML file, with each optional parameter specified with the default and required 
parameters left with a blank. Note that specifying :code:`null` is the same as leaving the parameter unspecified (i.e. :code:`None`).

::

    # Section 1: Simulation inputs
    gro:
    top:
    mdp:

    # Section 2: EEXE parameters
    parallel:
    n_sim:
    n_iter:
    s:
    nst_sim: null
    mc_scheme: 'metropolis' 
    w_scheme: null
    N_cutoff: 1000
    n_ex: 0
    runtime_args: null

    # Section 3: Output settings
    verbose: True
    n_ckpt: 100

    # Section 4: Data analysis
    msm: False
    free_energy: False 
    df_spacing: 1
    df_method: "MBAR"
    err_method: "propagate"
    n_bootstrap: 50
    seed : null

