.. _doc_basic_idea:

1. Basic idea
=============
Replica exchange of expanded ensemble (REXEE) integrates the core principles of replica exchange (REX)
and expanded ensemble (EE) methods.  Specifically, a REXEE simulation includes multiple
replicas of EE simulations in parallel and periodically exchanges coordinates
between replicas. Each replica samples a different but overlapping range of alchemical 
intermediate states to collectively sample the space bwteen the coupled (:math:`\lambda=0`)
and decoupled states (:math:`\lambda=1`). Naturally, the REXEE method decorrelates
the number of replicas from the number of states, allowing sampling a large number of intermediate
states with significantly fewer replicas than those required in the Hamiltonian replica exchange (HREX)
and other similar methods. By parallelizing replicas, the REXEE method also reduces
the simulation wall time compared to the EE method. More importantly, such parallelism sets the
stage for wider applications, such as relative free energy calculations for multi-topology transformations.

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


2. Replica swapping
===================

.. _doc_proposal:

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

4. Parameter space of REXEE
===========================