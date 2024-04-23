.. note:: This page is still a work in progress. Please check `Issue 41`_ for the current progress.

.. _`Issue 41`: https://github.com/wehs7661/ensemble_md/issues/41

.. _doc_basic_idea:

1. Basic idea
=============
Replica exchange of expanded ensembles (REXEE) [Hsu2024]_ integrates the core principles of replica exchange (REX)
and expanded ensemble (EE) methods.  Specifically, a REXEE simulation performs multiple
replicas of EE simulations in parallel and periodically exchanges coordinates
between replicas. Each replica samples a different but overlapping set of alchemical 
intermediate states to collectively sample the space bwteen the fully coupled (:math:`\lambda=0`)
and decoupled states (:math:`\lambda=1`). By design, the REXEE method decorrelates
the number of replicas from the number of states, enhancing the flexibility in replica configuration and 
allowing a large number of intermediate states to be sampled with significantly fewer replicas than those
required in the Hamiltonian replica exchange (HREX). By parallelizing replicas, the REXEE method also reduces
the simulation wall time compared to the EE method. More importantly, such parallelism sets the
stage for more complicated applications, especially one-shot free energy calculations that involve multiple
topologies, such as serial mutations or scaffold-hopping transformations.

In the following sections, we will briefly cover the theory behind the REXEE method, from its configuration, state
transitions, proposal schemes, weight combination schemes, to data analysis. For more details, please refer to the
paper [Hsu2024]_.

.. figure:: _static/REXEE_illustration.png
   :name: Figure 1
   :width: 800
   :align: center
   :figclass: align-center

   **Figure 1.** Schematic representation of the REXEE method, with the four configurational parameters annoated. In a REXEE simulation, the coordinates of replicas
   of EE simulations are periodically exchanged. :math:`{\bf A}_1`, :math:`{\bf A}_2`, :math:`{\bf A}_3`, and :math:`{\bf A}_4`
   denote the sets of states different replicas are sampling.

2. REXEE configuration
======================
Here we consider a REXEE simulation composed of :math:`R` parallel replicas of expanded ensembles, each of which is
labeled as :math:`i=1, 2, ..., R`. These :math:`R` replicas are restricted to sampling :math:`R` different yet overlapping
sets of states (termed **state sets**) labeled by :math:`m` as :math:`{\bf A}_1`, :math:`{\bf A}_2`, ..., :math:`{\bf A}_R`,
which collectively sample :math:`N` alchemical intermediate states in total, with :math:`N > R`. Additionally, we define :math:`s_i \in \{1, 2, ..., N\}`
as the index of the state currently sampled by the :math:`i`-th replica. For a replica :math:`i` sampling state set :math:`{\bf A}_m`,
:math:`s_i` is additionally constrained such that :math:`s_i \in {\bf A}_m`. Importantly, the fact that :math:`s_i` takes values
in :math:`\{1, 2, ..., N\}` and :math:`N>R` implies a many-to-one relationship between the replica index :math:`i` and the state index
:math:`s_i`, as a certain state may be sampled by multiple replicas. This is in contrast to the one-to-one relationship between the replica
index :math:`i` and the state set index :math:`m`, which ensures that each replica is associated with one and unique state set.

We emphasize that a valid REXEE configuration only requires overlapping state sets and is not restricted to one-dimensional grids,
the same number of states for all replicas, nor sequential state indices within the same state sets. For example, Figure 2 shows cases where
intermediate states are characterized by more than one thermodynamic variable (panels A and B), where different state sets
have different number of states (panels C), and where the state indices are not consecutive within the same state sets (panels A and C).
Still, the most common case is where the intermediate states are defined in a one-dimensional space, with consecutive state indices within
the same state set (e.g., the case in `Figure 1`_). In a REXEE simulation with such a configuration, a state shift :math:`\phi` between adjacent
state sets can be defined to indicate to what extent the set of states has shifted along the alchemical coordinate. Depending on whether all replicas
have the same number of states and whether or not the state shift is consistent between all adjacent state sets, a REXEE simulation can be either
homogeneous or heterogenous. Currently, :code:`ensemble_md` has only implemented the homogeneous REXEE method with one-dimensional alchemical intermediate
states defined sequentially in each state set.

.. figure:: _static/REXEE_more_configs.png
   :name: Figure 2
   :width: 800
   :align: center
   :figclass: align-center

   **Figure 2.** Different possible replica configurations of a REXEE simulation, with each state represented as a grid labeled by the number in its center
   and characterized by different Hamiltonians and/or temperatures. Different state sets are represented as dashed lines in different colors.
   Note that the temperature :math:`T` and Hamiltonian :math:`H` can be replaced by other physical variables of interest, such as pressure or chemical potential.

As shown in `Figure 1`_, a homogeneous REXEE simulation that samples sequential one-dimensional states can be configured by the following four parameters:

  - :math:`N`: The total number of intermediate states
  - :math:`R`: The total number of replicas
  - :math:`n_s`: The number of states per replica
  - :math:`\phi`: The state shift between adjacent state sets

These four configurational parameters are related via the following relationship:

.. math:: N = n_s + (R-1)\phi
   :label: eq_1

For example, the configuration of the REXEE simulation shown in `Figure 1`_ can be expressed as :math:`(N, R, n_s, \phi) = (9, 4, 6, 1)`. Importantly, the total
number of states :math:`N` does not have to be equal to the number of replicas :math:`R` in the REXEE method. In fact it is shown in the Supporting Information of
our paper [Hsu2024]_ that for a REXEE simulation simulation sampling any number of replicas, there exists at least one valid REXEE
configuration, allowing much higher flexibility in replica configuration compared to traditional replica exchange methods -- once the number of replicas
is decided, typically as a factor of the number of available cores, the total number of states can be arbitrary. In our Supporting Information, 
we also show that solving Equation :eq:`eq_1` with a few additional constraints allows efficient enumeration of all possible REXEE configurations. In :code:`ensemble_md`,
this enumeration is implemented in the command line interface (CLI) command :code:`explore_REXEE`, as elaborated in :ref:`doc_explore_REXEE`.

3. State transitions in REXEE simulations
=========================================
In a REXEE simulation, state transitions occur at both the intra-replica and inter-replica levels. Within each replica of expanded ensemble simulation,
transitions between alchemical states within the state set and the detailed balance conditions are governed by the selected algorithm in the expanded ensemble simulation
(i.e., the value of the GROMACS MDP parameter :code:`lmc-stats-move` in our implementation). Still, to ensure that probability influx and outflux are equal for each sets of states,
the detailed balance condition at the intra-replica level must be satisfied.

Mathematically, we consider replicas :math:`i` and :math:`j` that sample the state sets :math:`{\bf A}_m` and :math:`{\bf A}_n`, respectively. To swap replicas :math:`i`
and :math:`j`, the state sampled by replica :math:`i` at the moment, denoted as :math:`s_i \in {\bf A}_m`, must fall within the state set :math:`{\bf A}_n` that is to be swapped,
and vice versa. In this case, we call that these replicas :math:`i` and :math:`j` are **swappable**, and we express the exchange of coordinates :math:`x_i` and :math:`x_j` between these
two replicas as

.. math:: :label: eq_2
  
  X=\left(..., x^i_{m}, ..., x^j_{n}, ...\right) \rightarrow X' = \left(..., x^j_{m}, ..., x^i_{n}, ...\right)

with :math:`x^i_m \equiv (x_i, {\bf A}_m)` meaning that the :math:`i`-th replica samples the :math:`m`-th state set with the coordinates :math:`x_i`. Mathematically, the list of swappable pairs
:math:`\mathcal{S}` can be defined as the set of replica pairs as follows:

.. math:: :label: eq_3

  \mathcal{S} = \left\{(i, j) \mid s_i \in {\bf A}_n, s_j \in {\bf A}_m, i \neq j\right\}

As discussed in the Supporting Information of the paper [Hsu2024]_, the most straightforward way to derive the acceptance ratio that satisfies the intra-replica detailed balance condition 
is to assume symmetric proposal probabilities, which can be easily achieved by the design of the used proposal scheme. (See :ref:`doc_proposal` for more details.)
Under this assumption, the acceptance ratio of swapping the coordinates :math:`x_i` and :math:`x_j` between replicas :math:`i` and :math:`j` can be expressed as

.. math:: :label: eq_4

  P_{\text{acc}} = 
    \begin{cases} 
      \begin{aligned}
        &1 &, \text{if } \Delta \leq 0 \\
        \exp(&-\Delta) &, \text{if } \Delta >0
      \end{aligned}
    \end{cases}

where

.. math:: :label: eq_5

  \Delta = \left(u_{s_i}(x_j) + u_{s_j}(x_i) \right)-\left(u_{s_i}(x_i)+u_{s_j}(x_j)\right)

In Equation :eq:`eq_5`, :math:`u_{s_i}(x_j)` and :math:`u_{s_j}(x_i)` are the reduced potentials of the states :math:`s_i` and :math:`s_j` evaluated at the coordinates :math:`x_j` and :math:`x_i`, respectively.

.. _doc_proposal:

4. Proposal schemes
===================
In this section, we discuss proposal schemes available in the current implementation of the package :code:`ensemble_md`,
each of which has a symmetric proposal probability. These proposal schemes can be specified via the option :code:`proposal` in the input YAML file (e.g., :code:`params.yaml`)
for running a REXEE simulation. For more details about the input YAML file, please refer to :ref:`doc_parameters`.

- **Single exchange proposal scheme**:
  In this scheme, a pair of replicas is randomly drawn from the list of swappable pairs :math:`\mathcal{S}` defined in :eq:`eq_3`, with each pair in the list
  having an equal probability to be drawn. This method is the simplest and most straightforward proposal scheme, and it is the default proposal scheme in :code:`ensemble_md`.
- **Multiple exchange proposal scheme**:
  In this scheme, the number of swaps can be specified by the :code:`n_ex` parameter in the input YAML file. If :code:`n_ex` is not specified, :math:`N^3` swaps will be attempted in an exchange interval,
  where :math:`N` is the total number of alchemical intermediate states. For each attempted swap in this method, one pair will be drawn from the list of swappable pairs (with replacement). Between attempted swaps, the acceptance
  ratio is calculated to decide whether the swap should be accepted. Then, if the swap is accepted, the list of swappable pairs will be updated by re-identifying swappable pairs based on the updated permutation. (That is, the
  next attempted swap is dependent on whether the current swap is accepted.) If the swap is rejected, the execution will end and there won't be a new pair drawn. This method is more efficient than the single exchange proposal scheme
  as it allows multiple swaps to be attempted in a single exchange interval.

2.1. Single exchange proposal scheme
------------------------------------
The single exchange proposal scheme randomly draws a pair of replicas from the list of swappable pairs :math:`\mathcal{S}` defined in :eq:`eq_3`, with each pair in the list
having an equal probability to be drawn. 


This method can be used by spetting :code:`proposal: 'single'` in the input YAML file.


2.2. Multiple exchange proposal scheme
--------------------------------------


2.3. Exhaustive exchange proposal scheme
----------------------------------------



3. Correction schemes
=====================

3.1. Weight correction
----------------------




3.2. Histogram correction
-------------------------



4. Free energy calculations
===========================



2.1. Proposal schemes
---------------------
In the current implementation, the following proposal schemes are available and can be
specified via the option :code:`proposal` in the input YAML file (e.g. :code`params.yaml`).

2.1.1. Single swap
~~~~~~~~~~~~~~~~~~
If the option :code:`proposal` is specified as :code:`single` in the input YAML file, the method
of single swap will be used, in which only a single swap will be randomly drawn from the list of swappable
pairs within an exchange interval. Here, a pair of replicas can be swapped only if the states to be swapped 
are present in both of the alchemical ranges of the two replicas. After the pair is drawn, the acceptance
ratio will be calculated given the specified acceptance scheme to decide whether the swap should be accepted.

2.1.2. Neighboring swap
~~~~~~~~~~~~~~~~~~~~~~~
If the option :code:`proposal` is specified as :code:`neighboring` in the input YAML file, the method
of neighboring swap will be used. This method is exactly the same as the single swap method execpt that it
additionally requires the swappable pairs to be neighboring replicas.

2.1.3. Exhaustive swaps
~~~~~~~~~~~~~~~~~~~~~~~
If the option :code:`proposal` is specified as :code:`exhaustive` in the input YAML file, the method
of exhaustive swaps will be used. In an exchange interval, this method requires repeatedly drawing a pair
from the list of swappable pairs and remove pair(s) involving previously drawn replicas until the list is empty. 
For each proposed swap, we calculate the acceptance ratio to decide whether the swap should be accepted or not. 
In greater detail, this scheme can be decomposed into the following steps:
    
  - **Step 1**: Identify the list of swappable pairs. 
  - **Step 2**: Randomly draw a pair from the list of swappable pairs.
  - **Step 3**: Calculate the acceptance ratio for the drawn pair to decide whether the swap should be accepted.
    Then, perform or reject the swap. 
  - **Step 4**: Update the list of swappable pairs by removing pair(s) that involve any replica in the drawn pair in Step 2. 
  - **Step 5**: Repeat Steps 2 to 4 until the list of swappable pairs is empty.
  
Note that

  - In this method, no replicas should be involved in more than one proposed swap. 
  - Given :math:`N` alchemical intermediate states in total, one can at most perform :math:`\lfloor N \rfloor` swaps with this method.
  - While this method can lead to multiple attempted swaps, these swaps are entirely indepdent of each other, which is
    different from the method of multiple swaps introduced below.
  - Importantly, whether the swap in Step 3 is accepted or rejected does not influence the update of the list in Step 4 at all. 
    This is different from the method of multiple swaps introduced in the next section, where the updated list of swappable pairs depends on
    the acceptance/rejection of the current attempted swap. 
  - Since all swaps are independent, instead of calculating and acceptance ratio and performing swaps separately (as done in Step 3 in the procedure above), one
    can choose to calculates all acceptance ratios for all drawn pairs and perform all swaps at the same time at the end.
    We chose to implement the former in :obj:`.get_swapping_pattern` since this is more consistent with the protocol of the other proposal schemes
    , hence easier to code.

.. _doc_multiple_swaps:

2.1.4. Multiple swaps
~~~~~~~~~~~~~~~~~~~~~
If the option :code:`proposal` is specified as :code:`multiple` in the input YAML file, the method
of multiple swaps will be used, where the number of swaps can be specified by the :code:`n_ex` parameter in
the input YAML file. If :code:`n_ex` is not specified, :math:`N^3` swaps will be attmpted in an exchange interval,
where :math:`N` is the total number of alchemical intermediate states. For each attempted swap in this method, 
one pair will be drawn from the list of swappable pairs (with replacement). Between attempted swaps, the acceptance
ratio is calculated to decide whether the swap should be accepted. Then, if the swap is accepted, the list of 
swappable pairs will be updated by re-identifying swappable pairs based on the updated permutation. (That is, the
next attempted swap is dependent on whether the current swap is accepted.) If the swap is rejected, the execution will
end and there won't be a new pair drawn. In greater detail, this scheme can be decomposed into the following steps:

  - **Step 1**: Identify the list of swappable pairs.
  - **Step 2**: Randomly draw a pair from the list of swappable pairs.
  - **Step 3**: Calculate the acceptance ratio to decide whether the swap drawn in Step 2 should be accepted. 
  - **Step 4**: If the attempted swap is accepted, update the permutation of the state ranges and update the list of swappable
    pairs accordingly. Otherwise, the list of swappable pairs stay the same. 
  - **Step 5**: Perform Steps 2 and 3 for the desired number of times (specified by :code:`n_ex`).

.. note:: Except for the method of multiple swaps, all methods introduced above obey the detailed balance condition.
  We set the default of the paramter :code:`proposal` as :code:`exhaustive` as it allows higher sampling efficiency.

.. _doc_acceptance:

2.2. Acceptance schemes
-----------------------
In the current implementation, 3 acceptance schemes are available to be specified 
in the input YAML file (e.g. :code:`params.yaml`) via the option :code:`acceptance`, including :code:`same-state`/:code:`same_state`, 
and :code:`metropolis`. In our implementation, 
relevant methods include :obj:`.propose_swaps`, :obj:`.calc_prob_acc`, and :obj:`.accept_or_reject`.
Below we elaborate the details of each of the swapping schemes.

.. _doc_same_state:

2.2.1. Same-state swapping
~~~~~~~~~~~~~~~~~~~~~~~~~~
The simplest scheme for swapping replicas is the same-state swapping scheme, which only swaps 
replicas only if they both happen to same the same alchemical states right before the swap. That
is, the acceptance ratio is always either :math:`1` (same state) or :math:`0` (different states).
Notably, this swapping scheme does not obey the detailed balance condition.

2.2.2. Metropolis swapping 
~~~~~~~~~~~~~~~~~~~~~~~~~~
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

2.3. Calculation of Δ in Metropolis-based acceptance schemes
------------------------------------------------------------
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

2.3.1. The calculation of :math:`\beta[(U^i_n + U^j_m) - (U^i_m+U^j_n)]` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

In the case that mutiple swaps are desired (i.e. :code:`proposal` is :code:`multiple`), say :code:`n_ex` is 2, if the first swap between replicas 0 and 1 shown above is accepted and now 
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

2.3.2. The calculation of :math:`[(g^i_n + g^j_m) - (g^i_m+g^j_n)]` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

2.4. How is swapping performed?
-------------------------------
As implied in :ref:`doc_basic_idea`, in an REXEE simulation, we could either choose to swap configurations
(via swapping GRO files) or replicas (via swapping MDP files). In this package, we chose the former when
implementing the REXEE algorithm. Specifically, in the CLI :code:`run_REXEE`, the function :obj:`.get_swapping_pattern`
is called once for each iteration and returns a list :code:`swap_pattern` that informs :code:`run_REXEE` how
the GRO files should be swapped. (To better understand the list :code:`swap_pattern`, see the docstring of
the function :obj:`.get_swapping_pattern`.) Internally, the function :obj:`.get_swapping_pattern` not only swaps
the list :code:`swap_pattern` when an attempted move is accepted, but also swaps elements in lists that contains
state shifts, weights, paths to the DHDL files, state ranges, and the attribute :code:`configs`, but not the elements
in the list of states. Check the source code of :obj :`.get_swapping_pattern` if you want to understand the details.

.. _doc_w_schemes:

3. Weight combination
=====================

3.1. Basic idea
---------------
To leverage the stastics of the states collected from multiple replicas, we recommend 
combining the alchemical weights of these states across replicas during an weight-updating REXEE simulation.
Ideally, the modified weights should facilitate the convergence of the alchemical weights in expanded ensemble, 
which in the limit of inifinite simulation time correspond to dimensionless free energies of the alchemical states. 
The modified weights also directly influence the the accpetance ratio, hence the convergence of the simulation
ensemble. There are various possible ways to combine weights across replicas, though some of them might
have the issue of reference-dependent results, or the issue of coupling overlapped and non-overlapped states. 
Here, we present the most straightforward method that circumvent these two issues. This method can be enabled by
setting the parameter :code:`w_combine` to `True` in the input YAML file. 

3.2. The details of the method
------------------------------
Generally, weight combination is performed after the final configurations have beeen figured out and it is just for 
the initialization of the MDP files for the next iteration. Now, to demonstrate the method implemented in 
:code:`ensemble_md` (or more specifically, :obj:`.combine_weights`), here we consider the following sets of weights 
as an example, with :code:`X` denoting a state not present in the alchemical range:

::

    State       0         1         2         3         4         5      
    Rep A       0.0       2.1       4.0       3.7       X         X  
    Rep B       X         0.0       1.7       1.2       2.6       X    
    Rep C       X         X         0.0       -0.4      0.9       1.9

As shown above, the three replicas sample different but overlapping states. Now, our goal 
is to

* For state 1, combine the weights arcoss replicas 1 and 2.
* For states 2 and 3, combine the weights across all three replicas.
* For state 4, combine the weights across replicas 1 and 2. 

That is, we combine weights arcoss all replicas that sample the state of interest regardless 
which replicas are swapping. The outcome of the whole process should be three vectors of modified 
alchemical weights, one for each replica, that should be specified in the MDP files for the next iteration. 
Below we elaborate the details of each step carried out by our method.

Step 1: Calculate the weight difference between adjacent states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
First, we calculate the weight differences, which can be regarded rough estimates 
of free energy differences, between the adjacent states. We therefore have:

::

    States      (0, 1)    (1, 2)    (2, 3)    (3, 4)    (4, 5)    
    Rep 1       2.1       1.9       -0.3       X        X       
    Rep 2       X         1.7       -0.5       1.4      X       
    Rep 3       X         X         -0.4       1.3      1.0     

Note that to calculate the difference between, say, states 1 and 2, from a certain replica, 
both these states must be present in the alchemical range of the replica. Otherwise, a free 
energy difference can't not be calculated and is denoted with :code:`X`.

Step 2: Take the average of the weight differences across replicas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Then, for the weight differences that are available in more than 1 replica, we take the simple 
average of the weight differences. That is, we have:

::

    States      (0, 1)    (1, 2)    (2, 3)    (3, 4)    (4, 5)    
    Final       2.1       1.8       -0.4      1.35      1.0

Assigning the fist state as the reference, we have the following profile:

::
   
    Final g     0.0       2.1       3.9       3.5       4.85      5.85 

Notably, a weighted average is typically preferred as it is less sensitive to poor estimates. (See
the section of free energy calculations, where we use basically the same method as the one used here
except that weighte average are calculated.) However, a weighted average requires the uncertainties 
of the involved weight differences and calculating the uncertainties of the weight difference (which
basically are estimates of free energy differences) is to computationally expensive, so we only calculate
simple averages when combining the weights.

Step 3: Determine the vector of alchemical weights for each replica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Finally, we need to determine the vector of alchemical weights for each replica. To do this,
we just shift the weight of the first state of each replica back to 0. As a result, we have
the following vectors:

::

    State       0           1            2            3            4            5      
    Rep 1       0.0         2.1          3.9          3.5          X            X  
    Rep 2       X           0.0          1.8          1.4          2.75         X    
    Rep 3       X           X            0.0          -0.4         0.95         1.95

Again, as a reference, here are the original weights:

::

    State       0           1            2            3            4            5
    Rep 1       0.0         2.1          4.0          3.7          X            X
    Rep 2       X           0.0          1.7          1.2          2.6          X
    Rep 3       X           X            0.0          -0.4         0.9          1.9

Notably, taking the simple average of weight differences/free energy differences is equivalent to
taking the geometric average of the probability ratios.

.. _doc_histogram: 

3.3. Histogram corrections
--------------------------
In the weight-combining method shown above, we frequently exploted the relationship :math:`g(\lambda)=f(\lambda)=-\ln p(\lambda)`. 
However, this relationship is true only when the histogram of state vistation is exactly flat, which rarely happens in reality. 
To correct this deviation, we can convert the difference in the histogram counts into the difference in free energies. This is based 
on the fact that the ratio of histogram counts is equal to the ratio of probabilities, whose natural
logarithm is equal to the free energy difference of the states of interest. Specifically, we have:

.. math::
  g_k'=g_k + \ln\left(\frac{N_{k-1}}{N_k}\right)

where :math:`g_k'` is the corrected alchemical weight and :math:`N_{k-1}` and :math:`N_k` are the histogram 
counts of states :math:`k-1` and :math:`k`. 

Notably, this correction method can possibly overcorrect the weights when the histogram counts :math:`N_k` or :math:`N_{k-1}` are too low. 
To deal with this, the user can choose to specify :code:`N_cutoff` in the input YAML file, so that the the histogram
correction will performed only when :math:`\text{argmin}(N_k, N_{k-1})` is larger than the cutoff. Also, this histogram correction 
should always be carried out before weight combination. This method is implemented in :obj:`.histogram_correction`.
