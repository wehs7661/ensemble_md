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
In this section, we recommend a workflow of running an ensemble of expanded ensemble, which can be 
implmented by using functions defined :class:`ensemble_EXE`. 
A hands-on tutorial that implements this workflow can be found in `Tutorial 1: Ensemble of expanded ensemble`_. 

.. _`Tutorial 1: Ensemble of expanded ensemble`: examples/EEXE_tutorial.ipynb


Step 1: Set up parameters
-------------------------
To run an ensemble of expanded ensemble in GROMACS using :code:`ensemble_md`, one at 
least needs to following four files:

* One GRO file of the system of interest
* One TOP file of the system of interest
* One MDP template for customizing different MDP files for different replicas. 
* One YAML file that specify the EEXE-relevant parameters.

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

* Required parameters

  * :code:`parallel`: Whether the replicas of EEXE should be run in parallel or not.
  * :code:`n_sim`: The number of replica simulations.
  * :code:`n_iterations`: The number of iterations.
  * :code:`s`: The shift in the alchemical ranges between adjacent replicas (e.g. :math:`s = 2` if :math:`λ_2 = (2, 3, 4)` and :math:`λ_3 = (4, 5, 6)`.
  * :code:`mdp`: The MDP template that has the whole range of :math:`λ` values.

* Optional parameters

  * :code:`nst_sim`: The number of simulation steps, i.e. exchange frequency. This option assumes replicas with homogeneous simulation lengths. If this option is not specified, the number of steps defined in the template MDP file will be used. 
  * :code:`mc_scheme`: The method for swapping simulations. Choices include :code:`same-state`/:code:`same_state`, :code:`metropolis`, and :code:`metropolis-eq`/:code:`metropolis_eq`. For more details, please refer to :ref:`doc_mc_schemes`. (Default: :code:`metropolis`)
  * :code:`w_scheme`: The method for combining weights. Choices include :code:`None` (unspecified), :code:`exp-avg`/:code:`exp_avg`, and :code:`hist-exp-avg`/:code:`hist_exp_avg`. For more details, please refer to :ref:`doc_w_schemes`. (Default: :code:`hist-exp-avg`)
  * :code:`N_cutoff`: The histogram cutoff. Only required if :code:`hist-exp-avg` is used. (Default: 0)
  * :code:`n_ex`: The number of swaps to be proposed in one attempt. This works basically the same as :code:`-nex` flag in GROMACS. A recommended value is :math:`N^3`, where :math:`N` is the number of replicas. If `n_ex` is unspecified or specified as 0, neighboring swapping will be carried out. For more details, please refer to :ref:`doc_swap_basics`. (Default: 0)
  * :code:`outfile`: The output file for logging how replicas interact with each other. 
  * :code:`verbose`: Whether a verbse log is wanted. 

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


Replica swapping
================

.. _doc_mc_schemes:

MC schemes for swapping replicas
--------------------------------
In ensemble of expanded ensemble, we need to periodically exchange coordinates between 
replicas. Currently, we have implemented 3 Monte Carlo schemes for swapping replicas that can be specified 
in the input YAML file (e.g. :code:`params.yaml`) via the parameter :code:`mc_scheme`, including :code:`same-state`/:code:`same_state`, 
:code:`metropolis`, and :code:`metropolis-eq`/:code:`metropolis_eq`. In our implementation, 
relevant methods include :obj:`.propose_swaps`, :obj:`.calc_prob_acc`, and :obj:`.accept_or_reject`.
Below we elaborate the details of each of the swapping schemes.

.. _doc_same_state:

Same-state swapping
~~~~~~~~~~~~~~~~~~~
The simplest scheme for swapping replicas is the same-state swapping scheme, which only swaps 
replicas only if they both happen to same the same alchemical states right before the swap. That
is, the acceptance ratio is always either :math:`1` (same state) or :math:`0` (different states).
Notably, this swapping scheme does not obey the detailed balance condition.

Metropolis swapping 
~~~~~~~~~~~~~~~~~~~
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the limit of inifite simulation length, the alchemical weights of a certain 
alchemical (e.g. :math:`m` or :math:`n`) should be indepedent of the configuration
(e.g. :math:`i` or :math:`j`) being sampled, i.e. :math:`g^i_n=g^i_m` and :math:`g^j_n=g^j_m`. 
At this limit, where the weights are equilibrated, the expression of :math:`\Delta` in the
standard Metropolis swapping scheme reduces to the following:

.. math::
  \Delta = \beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)]

Notably, this scheme does not consider the difference in the alchemical weights, which can 
be non-zero frequently, so this swapping scheme does not strictly obey the detailed balance condition.

Calculation of :math:`\Delta` in Metropolis-based methods
---------------------------------------------------------
The calculation of :math:`\Delta` is important because the acceptance ratio :math:`w(X\rightarrow X')=\min(1, \exp(-\Delta))` is 
directly related to :math:`\Delta`. To better understand how :math:`\Delta` is calculated in the Metropolise-based methods, 
we need to first know what's available in the DHDL file of a GROMACS expanded ensemble simulation. As an example, below 
we tabulate the data of a DHDL file, with the time column having units of ps and all energy quantities having
units of kJ/mol. 

.. list-table::
   :widths: 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - Time
     - State
     - Total energy
     - :math:`dH/d\lambda` at :math:`\lambda_{\text{coul}}=0`
     - :math:`dH/d\lambda` at :math:`\lambda_{\text{vdw}}=0`
     - :math:`\Delta H` w.r.t :math:`(0, 0)`
     - :math:`\Delta H` w.r.t :math:`(0, 0.2)`
     - ...
   * - 0.0
     - 0
     - -16328.070	
     - 92.044243	
     - -24.358231	
     - 6.1035156e-05	
     - 18.408772
     - ...
   * - 2.0	
     - 1	
     - -16259.254	
     - 69.588318	
     - -5.8508954	
     - -13.917714	
     - -1.5258789e-05
     - ...
   * - 4.0
     - 7	
     - -16060.098	
     - -171.03197	
     - -55.529320	
     - 86505.967	
     - 86471.757
     - ...
   * - 6.0	
     - 6	
     - -16164.012	
     - -14.053808	
     - 30.875639	
     - 65.827232	
     - 63.016821
     - ...
   * - ...
     - ...
     - ...
     - ...
     - ...
     - ...
     - ...
     - ...

Notably, in the DHDL file, the total energy (i.e. Hamiltonian, denoted as :math:`H` in the table above) 
could be the sum of kinetic energy and potential energy or just the total potential energy, depending how the parameter 
:code:`dhdl-print-energy` is specified in the MDP file. However, as we will see later, we only care about 
:math:`\Delta H`, which should be equal to :math:`\Delta U` regardless of how :code:`dhdl-print-energy` is 
specified. This is because at each time frame, the kinetic energy of the system being at differnt :math:`\lambda`
values should be the same and cancelled out. (The kinetic energy is :math:`\lambda`-dependent.) With this, below 
we describe more details about the calculation of the difference in the potential energies and the difference in the alchemical weights.

The calculation of :math:`\beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)]` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Note that for each time frame shown in the table above, there is always one :math:`\Delta H` being 0, which happens when 
:math:`\Delta H` is calculated with respect to the state being sampled at that time frame. For example, if the vector of coupling 
parameters of state 6 is :math:`(0.5, 0)`, then at :math:`t=6` ps, when the replica is sampling state 6, :math:`\Delta H` w.r.t. :math:`(0.5, 0)` should be 0. This 
allows us to get the individual values of :math:`U^{i}_{n}`, :math:`U^{j}_{m}`, :math:`U^{i}_{m}`, and :math:`U^{j}_{n}` by assuming
the state being visited as the reference (i.e. :math:`U=0`). With this, we can calulcate :math:`\beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)]`
with ease. 

As an example, here we assume that we have four replicas labelled as 0, 1, 2, and 3 sampling configurations A, B, C and D and they 
ended up at states a, b, c, and d. Schematically, we have 
::

    replica       0       1       2       3
    state         a       b       c       d
    config        A       B       C       D

When swapping the configurations of replicas 0 and 1, we need to calculate the term :math:`\beta[(U^A_b + U^B_a) - (U^A_a+U^B_b)]`, or equivalently
:math:`\beta[(U^A_b - U^A_a) + (U^B_a-U^B_b)]`. Since now replica 0 is at state a at the end of the simulation, :math:`U^A_b - U^A_a` is immediately 
available in the DHDL file in replica 0, which is the final value of :math:`\Delta H` w.r.t :math:`(x, y)`, where :math:`(x, y)` 
is the vector of the coupling parameter of state b. 

Now let's say the table above comes from the DHDL file of replica 0. If at :math:`t=6` ps, we are swapping replicas A and B and 
:math:`a=6`, :math:`b=0` (i.e. at  :math:`t=6` ps, replicas A and B are sampling states 6 and 0, respectively), then :math:`U^A_b - U^A_a=U^A_0 - U^A_6=65.827232`.
Similarly, :math:`U^B_a - U^B_b` can be looked up in the DHDL file of replica 1, so :math:`\Delta` can be calculated. 
(Note that we need to convert the units of :math:`\Delta U` from kJ/mol to kT, which is more convenient for the calculation of the acceptance ratio.

In the case that mutiple swaps are desired, say :code:`n_ex` is 2, if the first swap between replicas 0 and 1 shown above is accepted and now 
we are swapping replicas 1 and 2 in the second swap, then we must be aware that the configurations now corresponding to replicas 0 and 1 is not A and B, 
but B and A, repsecitvely:
::

    replica       0       1       2       3
    state         a       b       c       d
    config        B       A       C       D

Therefore, when swapping replicas 1 and 2, instead of calculating :math:`\beta[(U^B_c - U^B_b) + (U^C_b-U^C_c)]`, we calculate :math:`\beta[(U^A_c - U^A_b) + (U^C_b-U^C_c)]`.
That is, when swapping replicas 1 and 2 in this case, instead of getting values from the DHDL files of replicas 1 and 2, we actually need to get values from 
the DHDL files of reaplicas 0 and 2. In this case, :math:`U^A_c - U^A_b` is not immediately available in the table because configuration A 
was at state :math:`a=6`, so the whole vector of :math:`\Delta H` is calculated against state 6. However, as mentioned above, we have 
:math:`U^A_6=0`, so we can still calculate :math:`U^A_c - U^A_b` by just taking the difference between :math:`\Delta H` w.r.t. :math:`(x_c, y_c)`
and :math:`\Delta H` w.r.t. :math:`(x_b, y_b)`, where :math:`(x_c, y_c)` and :math:`(x_b, y_b)` are the vectors of coupling parameters of states c and b. 
While all this process can sound a little confusing, it has been already taken care of by the function :obj:`.calc_prob_acc`.

The calculation of :math:`[(g^i_n + g^j_m) - (g^i_m+g^j_n)]` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the log file of a GROMACS expanded ensemble simulation, the alchemical weights have units of kT, which is why we don't have 
the inverse temperature :math:`\beta` multiplied with the weights. Unlike the potential energy terms, in the log file we can find 
the individual values of :math:`g^{i}_{n}`, :math:`g^{j}_{m}`, :math:`g^{i}_{m}` and :math:`g^{j}_{n}`. Now say that the log file of replica C 
reads below at :math:`t=6` ps:

:: 

              Step           Time
                500        6.00000

    Writing checkpoint, step 500 at Mon Jun 13 02:59:57 2022


                MC-lambda information
      Wang-Landau incrementor is:        0.32
      N  CoulL   VdwL    Count   G(in kT)  dG(in kT)
      1  0.000  0.000       18    0.00000    2.94000
      2  0.250  0.000       11    2.94000    1.26000
      3  0.500  0.000       13    4.20000    2.10000 <<
      4  0.750  0.000        2    6.30000    0.84000
      5  1.000  0.000        2    7.14000    0.04000
      6  1.000  0.250        4    7.18000    0.00000

Then apparently we have :math:`g^C_0=0` and :math:`g^C_6=7.18` kT, respectively. And the values of 
:math:`g^A_0` and :math:`g^A_6` can be found in the log file of replica A, which enables use to 
calculate :math:`(g^i_n+g^j_m)-(g^i_m+g^j_n)`. Notably, although it could be interesting to know 
the bias difference between different configurations sampling the same state in different alchemical 
ranges (i.e. :math:`g^i_n-g^j_n` and :math:`g^j_m-g^i_m`), it does not make sense to calculate such 
values from the log file because alchemical weights from in the log files corresponding to simulations 
sampling different alchemical ranges would have different references. Therefore, only values such as 
:math:`g^i_n-g^i_m` and :math:`g^j_m-g^j_n` make sense, even if they are as interesting as :math:`g^i_n-g^j_n` and :math:`g^j_m-g^i_m`.

.. _doc_w_schemes:

Weight combination
==================
As mentioned above, to leverage the stastics of the states collected from multiple replicas, we recommend 
combining the alchemical weights of these states across replicas to initialize the next iteration. Below 
we first describe how we shift weights to deal with the issue of different references of alchemical weights 
in GROMACS LOG files, then, we describe weight-combining methods available in our current implementation. 

Weight shifting
---------------
In the log file of a GROMACS expanded ensemble simulation, the alchemical weight of the first alchemical intermediate state 
is always shifted to 0 to serve as the reference. In EEXE, different replicas have different ranges of alchemical
states, i.e. different first states, hence difference references. For example, there could be 3 replicas having 
the following weights:

::

    State     0       1       2       3       4       5       6       7       8       9
    Rep 1     0.0     2.1     4.0     3.7     4.8     6.5     X       X       X       X
    Rep 2     X       X       0.0     -0.4    0.7     2.3     2.8     3.9     X       X
    Rep 3     X       X       X       X       0.0     1.5     2.1     3.3     6.0     9.2    

Each of these replicas sample 6 alchemical states. There alchemical ranges are different but overlapping. Specifically, 
Replicas 1 and 2 have overlap at states 2 to 5 and replicas 2 and 3 have overlap at states 4 to 7. Notably, all 
three replicas sampled states 4 and 5. Therefore, what we are going to do is

* For states 2 and 3, combine weights across replicas 1 and 2.
* For states 4 and 5, combine weights across replicas 1, 2 and 3.
* For states 6 and 7, combine weights across replicas 2 and 3.

However, before we combine the weights, we should make sure the weights of all replicas have the same reference because now 
the references of the 3 replicas are states 0, 2, and 4, respectively. Therefore could be 


Exponential averaging 
---------------------
In the limit that all alchemical states are equally sampled, the alchemical weight of a state 
is equal to the dimensionless free energy of that state, i.e. in the units of kT, we have 
:math:`g(\lambda)=f(\lambda)=-\ln p(\lambda)`, or :math:`p(\lambda)=\exp(-g(\lambda))`. Given this,
one intuitive way is to average the probability of the two simulations. Let :math:`g` be the weight combined from 
from the weights :math:`g_1` and :math:`g_2` and :math:`p`, :math:`p_1`, :math:`p_2` be the corresponding 
probabilities, then we have 

.. math::
  p = \frac{p_1+p_2}{2} = \frac{\text{e}^{-g_1} + \text{e}^{-g_2}}{2}

Given that :math:`p=\exp(-g)`, we have 

.. math::
  g = -\ln p = -\ln\left(\frac{\text{e}^{-g_1} + \text{e}^{-g_1}}{2} \right)

Exponential averaging with histogram corrections
------------------------------------------------
During the simulation, the histogram of the state visitation is generally not flat, such that
:math:`g` is no longer equal to the dimensionless free energy, i.e. :math:`g(\lambda)=-\ln p(\lambda)`
is no longer true. However, the ratio of counts is equal to the ratio of probabilities, whose natural
logarithm is equal to the free energy difference of the states of interest. With this, we can do
the following corrections before combining the weights:

.. math::
  g_k'=g_k + \ln\left(\frac{N_{k-1}}{N_k}\right)

where :math`g_k'` is the corrected alchemical weight and :math:`N_{k-1}` and :math:`N_k` are the histogram 
counts of states :math:`k-1` and :math:`k`. Combining this correction with exponential averaging, we have 

.. math::
  g =  -\ln\left(\frac{\text{e}^{-g_1'} + \text{e}^{-g_2'}}{2} \right)

where :math:`g_1'` and :math:`g_2'` are weights corrected based on their histogram counts. 

Notably, this histogram correction should be carried out before shifting the weights, so the workflow we be
first correcting the weights, shifting the weights, and finally combining the weights. Also, this correction 
method can be overcorrect the weights when the histogram counts :math:`N_k` or :math:`N_{k-1}` are too low. 
To deal with this, the user can choose to specify :code:`N_cutoff` in the input YAML file, so that the the histogram
correction will performed only when :math:`\text{argmin}(N_k, N_{k-1})` is larger than the cutoff, otherwise this method 
will reduce to the standard exponential averaging method. 
