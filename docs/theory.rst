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

Basic idea
----------
As mentioned above, to leverage the stastics of the states collected from multiple replicas, we recommend 
combining the alchemical weights of these states across replicas to initialize the next iteration. Ideally,
well-modified weights should facilitate the convergence of the alchemical weights in expanded ensemble, which 
in the limit of inifinite simulation time correspond to dimensionless free energies of the alchemical states. 
The modified weights also directly influence the the accpetance ratio, hence the convergence of the simulation
ensemble. Potentially, there are various methods for combining weights across multiple replicas. One intuitive 
method to average the probabilities :math:`p_1` and :math:`p_1` that respectively correspond to weights :math:`g_1` 
and :math:`g_1`, i.e. 

.. math::
  g=\ln p = -\ln\left(\frac{p_1+p_2}{2}\right) = -\ln\left(\frac{\text{e}^{-g_1} + \text{e}^{-g_2}}{2}\right)

This exploits the fact that in expanded ensemble, the alchemical weight of a state is the dimensionless free energy
of that state given an exactly flat histogram of state visitation. While this assumption of flat histograms is generally 
not true, espeically in cases where the free energy differen of interest is large, one can consider "correcting"
the weights before combining them. (See :ref:`doc_histogram` for more details.)

While the method illustrated above is intuitive and easy to operate, it suffers from the issue of reference state selection.
This issue comes from the fact that GROMACS always shifts the weight of the first state to 0 to make it as the reference state.
Given that the first state of different replicas in EEXE are different, this essentially means that the vector of 
alchemical weights of different replicas have different references. Although it is possible to pick a reference 
state for all replicas before weight combination could solve this issue, different choices of references could lead to 
slightly different combined weights, hence probability ratios. As there is no real justification which state should be favored
as the reference, instead of the method explained above, we implemented another method that exploits the average of "probability ratios"
to circumvent the issue of reference selection. 

Weight combinination based on probability ratios
------------------------------------------------
Generally, weight combination is performed after the final configurations have beeen figured out and it is just for 
the initialization of the MDP files for the next iteration. Now, to demonstrate the method implemented in 
:code:`ensemble_md` (or more specifically, :obj:`.combine_weights`, here we consider the following sets of weights 
as an example, with :code:`X` denoting a state not present in the alchemical range:

::

    State       0         1         2         3         4         5      
    Rep 1       0.0       2.1       4.0       3.7       X         X  
    Rep 2       X         0.0       1.7       1.2       2.6       X    
    Rep 3       X         X         0.0       -0.4      0.9       1.9

As shown above, the three replicas sample different but overlapping states. Now, our goal 
is to

* For state 1, combine the weights arcoss replicas 1 and 2.
* For states 2 and 3, combine the weights across all three replicas.
* For state 4, combine the weights across replicas 1 and 2. 

That is, we combine weights arcoss all replicas that sample the state of interest regardless 
which replicas are swapping. The outcome of the whole process should be three vectors of modified 
alchemical weights, one for each replica, that should be specified in the MDP files for the next iteration. 
Below we elaborate the details of each step carried out by our method.

Step 1: Convert the weights into probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For weight :math:`g_ij` that corresponds to state :math:`j` in replica :math:`i`, we can calculate its 
corresopnding probability as follows:

.. math::
  p_{ij}=\frac{\exp(-g_{ij})}{\sum_{j=1}^N \exp(-g_{ij})}

where :math:`N` is the number of states in replica :math:`i`. As a result, we have the following probabilities
for each replica. Note that the sum of the probabilities of each row (replica) should be 1.

::

    State      0            1            2            3            4          5      
    Rep 1      0.85800      0.10507      0.01571      0.02121      X          X  
    Rep 2      X            0.64179      0.11724      0.19330      0.04767    X   
    Rep 3      X            X            0.32809      0.48945      0.13339    0.04907


Step 2: Calculate the probability ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ideally (in the limit of inifinite simulation time), for the 3 states overlapped between replicas 1 and 2, 
we should have

.. math::
    r_{2, 1} = \frac{p_{2i}}{p_{1i}} = \frac{p_{21}}{p_{11}} = \frac{p_{22}}{p_{12}}= \frac{p_{23}}{p_{13}} 

where :math:`r_{2, 1}` is the "probability ratio" between replicas 2 and 1. However, the probability ratios 
corresopnding to different states will not be the same in practice, but will diverge with statistical noise
for short timescales. For example, in our case we have the following ratios. (Note that here we calculate with
full precision but only present a few significant figures.)

.. math::
    \frac{p_{21}}{p_{11}}=6.10828, \; \frac{p_{22}}{p_{12}} = 7.46068, \; \frac{p_{23}}{p_{13}}=9.11249

Similarly, for states 2 to 4, we need to calculate the probability ratios between replicas 2 and 3:

.. math::
    \frac{p_{32}}{p_{22}}=2.79834, \; \frac{p_{33}}{p_{23}} = 2.53204, \; \frac{p_{34}}{p_{24}}=2.79834

Notably, in this case, there is no need to calculate :math:`r_{3, 1}` because :math:`r_{3, 1}` is already determined
as :math:`r_{3, 1}=r_{3, 2} \times r_{2, 1}`. Also, there are only 2 overlapping states between replicas 1 and 3,
but we want to maximize the overlap when combining weights. Therefore, the rule of thumb of calculating the 
probability ratios is that we only calculate the ones betwee adjacent replicas, i.e. :math:`r_{i+1, i}`.


Step 3: Average the probability ratios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now, to determine an unifying probability ratio between a pair of replicas, we can choose to take simple averages 
or geometric averages. 

- Method 1: Simple average

.. math::
    r_{2, 1} = \frac{1}{3}\left(\frac{p_{21}}{p_{11}} + \frac{p_{22}}{p_{12}} + \frac{p_{23}}{p_{13}} \right) \approx 7.56049, \;
    r_{3, 2} = \frac{1}{3}\left(\frac{p_{32}}{p_{22}} + \frac{p_{33}}{p_{23}} + \frac{p_{34}}{p_{24}} \right) \approx 2.70957

- Method 2: Geometric average

.. math::
    r_{2, 1}' = \left(\frac{p_{21}}{p_{11}} \times \frac{p_{22}}{p_{12}} \times \frac{p_{23}}{p_{13}} \right)^{\frac{1}{3}} \approx 7.46068, \;
    r_{3, 2}' = \left(\frac{p_{32}}{p_{22}} \times \frac{p_{33}}{p_{23}} \times \frac{p_{34}}{p_{24}} \right)^{\frac{1}{3}} \approx 2.70660

Notably, if we take the negative natural logarithm of a probability ratio, we will get a free energy difference. For example, 
:math:`-\ln (p_{21}/p_{11})=f_{21}-f_{11}`, i.e. the free energy difference between state 1 in replica 2 and state 1 in replica 1. 
(This value is generally not 0 because different replicas have different references.) Therefore, while Method 1 takes the simple 
average the probability ratios, Method 2 essentially averages such free energy differences. Both methods are valid in theory and 
should not make a big difference in the convergence speed of the simulations because we just need an estimate of free energy for each 
state better than the weight of a single simulation. In fact, the closer the probability ratios are from each other, the closer 
the simple average is from the geometric average. 

Step 4: Scale the probabilities for each replica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the simple averages of the probability ratios :math:`r_{21}` and :math:`r_{32}`, we can scale the probability vectors of 
replicas 2 and 3 as follows:

::

    State       0            1            2            3            4            5      
    Rep 1       0.85800      0.10507      0.01571      0.02121      X            X  
    Rep 2’      X            0.08489      0.01551      0.02557      0.00630      X   
    Rep 3’      X            X            0.01602      0.02389      0.00651      0.00240
  
As shown above, we keep the probability vector of replica 1 the same but scale that for the other two. Specifically, the probability 
vector of replica 2' is that of replica 2 divided by :math:`r_21` and the probability vector of replica 3' is that of replica 3 
divided by :math:`r_{21} \times r_{32}`. 

Similarly, if we used the probability ratios :math:`r_{21}'` and :math:`r_{32}'`, we would have had:

::

    State       0            1            2            3            4            5      
    Rep 1       0.85800      0.10507      0.01614      0.02121      X            X  
    Rep 2’      X            0.08602      0.01571      0.02591      0.00639      X   
    Rep 3’      X            X            0.01625      0.02424      0.00661      0.00243 


Step 5: Average and convert the probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After we have the scaled probabilities, we need to average them for each state, with the averaging method we used 
to average the probability ratios. For example, for state 1, we need to calculate the simple average of 0.105 and 0.08529, 
or the geometric average of 0.105 nad 0.08529. As such, for the first case (where the probabilities were scaled with :math:`r_{21}` 
and :math:`r_{32}`), we have the following probability vector of full range:

:: 

    Final p      0.85800      0.09498      0.01575      0.02356      0.00641      0.00240

which can be converted to the following vector of alchemical weights  (denoted as :math:`\vec{g}`) by taking the negative natural logarithm:

::

    Final g      0.15321      2.35412      4.15117      3.74831      5.05019      6.03420

For the second case (scaled with :math:`r_{21}'` and :math:`r_{32}'`), we have 

::

    Final p      0.85800      0.09507      0.01589      0.02371      0.00649      0.00243

which can be converted to the following vector of alchemical weights (denoted as :math:`\vec{g}'`):

::

    Final g      0.15315      2.35314      4.14203      3.74204      5.03658      6.01981


Step 6: Determine the vector of alchemical weights for each replica
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lastly, with the the vector of alchemical weights of the full range, we can figure out the alchemical weights 
for each replica, by shifting the weight of the first state of each replica back to 0. That is, with :math:`\vec{g}`,
we have:

::

    State      0            1            2           3             4           5      
    Rep 1      0.00000      2.20097      3.99802     3.59516       X           X  
    Rep 2      X            0.00000      1.79706     1.39419       2.69607     X   
    Rep 3      X            X            0.00000     -0.40286      0.89901     1.88303 

Similarly, with :math:`\vec{g}'`, we have:

::

    State      0            1            2            3            4            5      
    Rep 1      0.00000      2.20000      3.98889      3.58889      X            X  
    Rep 2      X            0.00000      1.78889      1.38889      2.68333      X   
    Rep 3      X            X            0.00000      -0.40000     0.89444      1.87778 

As a reference, here are the original weights:

::

    State       0           1            2            3            4            5
    Rep 1       0.0         2.1          4.0          3.7          X            X
    Rep 2       X           0.0          1.7          1.2          2.6          X
    Rep 3       X           X            0.0          -0.4         0.9          1.9

As shown above, the results using Method 1 and Method 2 are pretty close to each other. Notably, regardless of 
which type of averages we took, in the case here we were assuming that each replica is equally weighted. In the 
future, we might want to assign different weights to different replicas such that the uncertainty of free energies
can be minimized. For example, if we are combining probabilities :math:`p_1` and :math:`p_2` that respectively 
have uncertainties :math:`\sigma_1` and :math:`\sigma_2`, we can have 

.. math::
    p = \left(\frac{\sigma_2}{\sigma_1 + \sigma_2}\right)p_1 + \left(\frac{\sigma_1}{\sigma_1 + \sigma_2}\right)p_2

However, calculating the uncertainties of the :math:`p_1` and :math:`p_2` on-the-fly is generally difficult, so 
this method has not been implemented. 

.. _doc_histogram: 

Histogram corrections
---------------------
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
