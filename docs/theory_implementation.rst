.. _doc_basic_idea:

Basic idea
==========
Ensemble of expanded ensemble (EEXE) integrates the core principles of replica exchange 
molecular dynamics (REMD) and expanded ensemble (EXE).  Specifically, an ensemble of 
expanded ensembles includes multiple non-interacting, parallel expanded ensemble simulations 
that collectively sample a number of alchemical states spanning between the coupled state 
(:math:`\lambda=0`) and the uncoupled state (:math:`\lambda=1`). Each expanded ensemble 
samples a subset of these states such that the range of its allowed alchemical states 
overlaps with that of the adjacent replicas. In EEXE, the exchange of coordinates/alchemical 
states occurs at a specified frequency, which is beneficial for allowing better mixing 
in the alchemical space given sufficiently long simulation time, properly specified parameters 
and highly parallelizable computing architectures. 

Mathematically, we first consider a simulation ensemble that consists of :math:`M` non-interacting replicas 
of the expanded ensemble simulations all at the same constant temperature :math:`T`. We assume 
that the expanded ensembles collectively sample :math:`N` (:math:`N < M`) alchemical states with 
labels :math:`m=1, 2, ..., N` and each replica sampling :math:`n_i` states starting from state 
:math:`k` to state :math:`k + n_i -1`, which correspond to :math:`\lambda` vectors :math:`\lambda_k`, 
:math:`\lambda_{k+1}`, ..., and :math:`\lambda_{k+n_i-1}`, repsectively. Mathematically, we can 
define a non-injective but surjective function :math:`f` that maps the labels for replicas 
(:math:`i`) to labels for :math:`\lambda` vectors (:math:`m`): 

.. math::
   m=m(i) \equiv f(i)

This essentially assumes that the discrete domain  :math:`\left \{k, k+1, ..., k+n_i-1 \right \}` 
is always a subset of :math:`\mathcal{N} = \left \{1, 2, ..., N \right \}`. Notably, this is 
different from Hamiltonian replica exchange molecular dynamics (HREMD) or temperature replica exchange 
molecular dynamics (TREMD), where the mapping function should always be bijective (i.e. injective and 
surjective) such that :math:`i=i(m) \equiv f(m)` and :math:`m=m(i) \equiv f^{-1}(i)`. That is, in HREMD 
and TREMD, the number of alchemical states/temperatures is always the same as the number of replicas 
(surjective) and there is always a one-to-one correpsondence between the two (injective). Physically, 
this means that while in HREMD and TREMD, exchanging a pair of replicas is equivalent to exchanging 
a pair of temperatures :math:`\lambda` vectors, we don't regard exchanging replicas as exchanging :math:`\lambda`
vectors because it is a many-to-one correpsondence instead of one-to-one correspondence.

Now, we can write the "reduced Hamiltonian" of the :math:`i`-th expanded ensemble as 

.. math::
  H(p^{i}, q^{i}, \lambda_{m}^{i}) = k(p^{i}) + u(q^{i}, \lambda_{m}^{i})

where the reduced kinetic energy :math:`k(p^{i})` and reduced potential 
:math:`u(q^{i}`, :math:`\lambda_{m}^{i})` can be expressed as 

.. math::
  \begin{cases} 
  k(p^i) = \beta K(p^{i}) \\       
  u(q^i, \lambda^{i}_{m}) = \beta U(q^{i}, \lambda^{i}_{m}) - g^{i}_{m}\\
  \end{cases}

with :math:`\beta=1/kT` being the inverse temperature and :math:`g^{i}_{m}` being the weighting factor 
applied to the :math:`m`-th alchemical state (:math:`m \in \mathcal{N}`) of 
the :math:`i`-th replica. As such, the probability of the state being sampled (say, state :math:`m`) 
can be written as:

.. math::
  p = \frac{\exp(-\beta H(p^{i}, q^{i}))}{\int \exp(-\beta H(p^{i}, q^{i})) dq^{i}}=\frac{\exp(-\beta K(p^{i}) -\beta U(q^i, \lambda^{i}_{m}) + g^{i}_{m})}{\int \exp(-\beta K(p^{i}) -\beta U(q^i, \lambda^{i}_{m}) + g^{i}_{m}) dq^{i}}

Let :math:`X=(x^{1}_{m(1)}, x^{2}_{m(2)}, ..., x^{M}_{m(M)})` stand for a "state" in the generalized ensemble 
sampled by EXEE, where the superscript and subscript of :math:`x^{i}_{m(i)}` are the labels for the 
replica and :math:`\lambda` states, respectively. The state :math:`X` is specified by the :math:`M` sets of 
coordinates :math:`q^{i}` and momenta :math:`q^{i}` in replica :math:`i` with a :math:`\lambda` vector 
:math:`\lambda^{i}_{m}` such that :math:`x^{i}_{m}\equiv(q^{i}, p^{i})_m`. Given that the replicas of 
expanded ensemble are non-interacting, the weight factor for a state :math:`X` in this generalized ensemble 
is given by the product of Boltzmann factors for each replica:

.. math::
  W(X) = \exp \left [\sum^{M}_{i=1} (-\beta K(p^{i}) -\beta U(q^i, \lambda^{i}_{m}) + g^{i}_{m})\right ]

Now, we consider exchanging replicas of expanded ensemble :math:`i` and :math:`j`:

.. math::
  X = (..., x_{m}^{i}, ..., x_{n}^{j}, ...) \rightarrow X' = (..., x_{f'(i)}^{i}, ..., x_{f'(j)}^{j}, ...) = (..., x_{n}^{i}, ..., x_{m}^{j}, ...)

Notably, such an exchange introduces a new non-injective but surjective function :math:`f'` mapping the label 
:math:`i` to the label :math:`m`. (Note that since both functions :math:`f` or :math:`f'` are not bijective, 
we don't have their inverse to map the label :math:`m` back to the label :math:`i`.) That is, we have:

.. math::
  m = f(i) \rightarrow n=f'(i)

To ensure the convergence of the exchange process towards an equilibrium distribution, we impose the detailed 
balance condition on the transition probability :math:`w(X \rightarrow X')`:

.. math::
  W(X)w(X \rightarrow X') = W(X')w(X' \rightarrow X)

Here, we introduce a shorthand expression for the potential energy such that terms like :math:`U(q^i, \lambda^{i}_{m})` 
can be rewriteen as :math:`U^i_m`. With this, we have

.. math::
  \begin{aligned}
  \frac{w(X \rightarrow X')}{w(X' \rightarrow X)} & = \frac{W(X')}{W(X)} \\
          & = \frac{\exp(-\beta K(p^{i}) -\beta U(q^{i}, \lambda^{i}_{n}) + g^{i}_{n} -\beta K(p^{j}) -\beta U(q^{j}, \lambda^{j}_{m}) + g^{j}_{m})}{\exp(-\beta K(p^{i}) -\beta U(q^{i}, \lambda^{i}_{m}) + g^{i}_{m} -\beta K(p^{j}) -\beta U(q^{j}, \lambda^{j}_{n}) + g^{j}_{n})} \\
          & = \exp(-\beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)] + [(g^i_n+g^j_m)-(g^i_m+g^j_n)]) \\
  \end{aligned}
  \label{trans}

where

.. math::
  \Delta = \beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)] - [(g^i_n+g^j_m)-(g^i_m+g^j_n)] 

Notably, in the equation above, all kinetic energy terms and the Hamiltonians of non-exchanging 
replicas are canceled out. Now, using the usual Metropolis criterion, the transition probability 
:math:`w(X \rightarrow X')` can be expressed as

.. math::
  w(X \rightarrow X') = 
  \begin{cases} 
    \begin{aligned}
      &1 &, \;\text{if} \;\Delta \leq 0 \\
      \exp(&-\Delta) &, \;\text{if} \;\Delta >0
    \end{aligned}
  \end{cases}

Notably, if the systems of the replicas to be exchanged are sampling the same alchemical 
state (namely, :math:`m=n`) right before the exchange occurs, :math:`\Delta` will reduce to 
0, meaning the the exchange will always be accepted. 

Suggested workflow
==================
In this section, we recommend a workflow of running an ensemble of expanded ensemble. 
A hands-on tutorial that implements this workflow can be found in `Tutorial 1: Ensemble of expanded ensemble`_. 

.. _`Tutorial 1: Ensemble of expanded ensemble`: examples/EEXE_tutorial.ipynb


Step 1: Set up parameters
-------------------------
To run an ensemble of expanded ensemble in GROMACS using :code:`ensemble_md`, one at 
least needs to following four files:

  - One GRO file of the system of interest
  - One TOP file of the system of interest
  - One MDP template for customizing different MDP files for different replicas. 
  - One YAML file that specify the EEXE-relevant parameters.

Notably, here we are assuming that all replicas start from the same configuration represented 
by the single GRO file, but the user should also be able to use the methods defined in 
:code:`ensemble_md` to initialize different replicas with different configurations (represented by
multiple GRO files) in the first iteration. Also, the MDP template should contain parameters 
common across all replicas and define the coupling parmaeters for all possible intermediate states,
so that we can cusotmize different MDP files by defining a subset of alchemical states in different 
replicas. 

Importantly, to instantiate the class :class:`.EnsembleEXE`, the input YAML file should be passed.
In this YAML file, the user needs to specify how the replicas should be set up or interact with each 
other during the simulation ensemble. Below we decribe the details of these parameters.

  - Required parameters

    - :code:`parallel`: Whether the replicas of EEXE should be run in parallel or not.
    - :code:`n_sim`: The number of replica simulations.
    - :code:`n_iterations`: The number of iterations.
    - :code:`s`: The shift in the alchemical ranges between adjacent replicas (e.g. :math:`s = 2` if :math:`λ_2 = (2, 3, 4)` and :math:`λ_3 = (4, 5, 6)`.
    - :code:`mdp`: The MDP template that has the whole range of :math:`λ` values.

  - Optional parameters

    - :code:`nst_sim`: The number of simulation steps, i.e. exchange frequency. This option assumes replicas with homogeneous simulation lengths. If this option is not specified, the number of steps defined in the template MDP file will be used. 
    - :code:`mc_scheme`: The method for swapping simulations. Choices include :code:`same-state`/:code:`same_state`, :code:`metropolis`, and :code:`metropolis-eq`/:code:`metropolis_eq`. For more details, please refer to :ref:`doc_mc_schemes`. (Default: :code:`metropolis`)
    - :code:`w_scheme`: The method for combining weights. Choices include :code:`None` (unspecified), :code:`exp-avg`/:code:`exp_avg`, and :code:`hist-exp-avg`/:code:`hist_exp_avg`. For more details, please refer to :ref:`doc_w_schemes`. (Default: :code:`hist-exp-avg`)
    - :code:`N_cutoff`: The histogram cutoff. Only required if :code:`hist-exp-avg` is used. (Default: 0)
    - :code:`n_pairs`: The number of pairs of simulations to be swapped in each attempt. Note that the maximum number of :code:`n_pairs` is half of :code:`n_sim`
    - :code:`outfile`: The output file for logging how replicas interact with each other. 

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

  - Find the last sampled state and the corresponding lambda values from the DHDL files
  - Find the final Wang-Landau incrementors and weights from the LOG files. 

These two tasks are done by :obj:`.extract_final_dhdl_info` and :obj:`.extract_final_log_info`.

Step 3-2: Identify swappable pairs and propose simulation swap(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After the information of the final status of the previous iteration is extracted, we then identify swappable pairs.
Specifically, replicas can be swapped only if the states to be swapped are present in both of the alchemical ranges 
corresponding to the two replicas. This definition inherently implies one necessary but not sufficient condition that 
the replicas to be swapped should have overlapping alchemical ranges. Practically, if the states to be swapped are 
not present in both alchemical ranges, information like :math:`\Delta U^i=U^i_n-U^j_m` will not be available 
in either DHDL files and terms like :math:`\Delta g^i=g^i_n-g^i_m` cannot be calculated from the LOG files as well, which 
makes the calculation of the acceptance ratio technicaly impossible. (For more details about the acceptance ratio is calculated
in different schemes for swapping, check the section :ref:`doc_mc_schemes`.) After the swappable pairs are identified, 
the user can propose swap(s) using :obj:`propose_swaps`. 

Step 3-3: Decide whether to reject/accept the swap(s)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This step is mainly done by :obj:`.calc_prob_acc` and :obj:`.accept_or_reject`. The former calculates the acceptance 
ratio from the DHDL/LOG files of the swapping replicas, while the latter draws a random number and compare with the 
acceptance ratio to decide whether the proposed swap should be accepted or not.

Step 3-4: Combine the weights if needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the states that are present in the alchemical ranges of multiple replicas, it is likely that they are 
sampled more frequenly overall. To leverage the fact that we collect more statistics for these states, it is recoomended 
that the weights of these states be combined across all replicas that sampled these states. This task can be completed by
:obj:`combine_wieghts`, with the desired method specified in the input YAML file. For more details about different 
methods for combining weights across different replicas, please refer to the section :ref:`doc_w_schemes`.


Step 3-5: Modify the MDP files and swap out the GRO files (if needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once it has been figured out whether there is a pair of simulations to be swapped, the user should set up the input 
fils for the next iteration. In principle, the new iteration should inherit the final status of the previous iteration. 
This means:

  - For each replica, the input configuration for initializing a new iterations should be the output configuraiton of the previous iteration. If replicas :math:`i` and :math:`j` should be swapped, the new iteration of replica :math:`i` should be intialized by the output configuration of :math:`j` and vice versa. (Instead of exchanging the mdp files, we recommend swapping out the coordinate files to exchange replicas.)
  - For each replica, the MDP file for the new iteration should be the same as the one used in the previous iteartion except that parameters like :code:`tinit`, :code:`init-lambda-state`, :code:`init-wl-delta`, and :code:`init-lambda-weights` should be modified to the final values in the previous iteration. This can be done by :class:`.gmx_parser.MDP` and :obj:`.update_MDP`.



Step 4: Run the new iteration
-----------------------------
After the input files for a new iteration have been set up, we use the procedure in Step 2 to 
run a new iteration. Then, the user should loop between Steps 3 and 4 until the desired number of 
iterations (:code:`n_iterations`) is reached. 

.. _doc_mc_schemes:

MC schemes for swapping replicas
================================
In ensemble of expanded ensemble, we need to periodically exchange coordinates between 
replicas. Currently, we have implemented 3 Monte Carlo schemes for swapping replicas that can be specified 
in the input YAML file (e.g. :code:`params.yaml`) via the parameter :code:`mc_scheme`, including :code:`same-state`/:code:`same_state`, 
:code:`metropolis`, and :code:`metropolis-eq`/:code:`metropolis_eq`. In our implementation, 
relevant methods include :obj:`.propose_swaps`, :obj:`.calc_prob_acc`, and :obj:`.accept_or_reject`.
Below we elaborate the details of each of the swapping schemes.

.. _doc_same_state:

Same-state swapping
-------------------
The simplest scheme for swapping replicas is the same-state swapping scheme, which only swaps 
replicas only if they both happen to same the same alchemical states right before the swap. That
is, the acceptance ratio is always either :math:`1` (same state) or :math:`0` (different states).
Notably, this swapping scheme does not obey the detailed balance condition.

Metropolis swapping 
-------------------
Metropolis swapping uses the Metropolis criterion to swap replicas, i.e. 

.. math::
  w(X \rightarrow X') = 
  \begin{cases} 
    \begin{aligned}
      &1 &, \;\text{if} \;\Delta \leq 0 \\
      \exp(&-\Delta) &, \;\text{if} \;\Delta >0
    \end{aligned}
  \end{cases}

where 

.. math::
  \Delta = \beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)] - [(g^i_n+g^j_m)-(g^i_m+g^j_n)] 

In theory, this swapping scheme should obey the detailed balance condition, as derived 
in :ref:`doc_basic_idea`.

Equilibrated Metropolis swapping
--------------------------------
In the limit of inifite simulation length, the alchemical weights of a certain 
alchemical (e.g. :math:`m` or :math:`n`) should be indepedent of the configuration
(e.g. :math:`i` or :math:`j`) being sampled, i.e. :math:`g^i_n=g^i_m` and :math:`g^j_n=g^j_m`. 
At this limit, where the weights are equilibrated, the expression of :math:`\Delta` in the
standard Metropolis swapping scheme reduces to the following:

.. math::
  \Delta = \beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)]

Notably, this scheme does not consider the difference in the alchemical weights, which can 
be non-zero frequently, so this swapping scheme does not strictly obey the detailed balance condition.

.. _doc_w_schemes:

Weight combination
==================
When exchanging replicas in ensemble of expanded ensemble, we can swap 
out the coordinates of the replicas, 




Weight shifting
---------------

Exponential averaging 
---------------------

Exponential averaging with histogram corrections
------------------------------------------------

Transition matrix
=================
Theoretical and experimental transition matrix 
----------------------------------------------

State transition matrix
-----------------------

Replica transition matrix
-------------------------

Free energy calculation
=======================
