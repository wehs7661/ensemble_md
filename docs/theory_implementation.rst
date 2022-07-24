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

Acceptance ratio
================
Here, we first consider a simulation ensemble that consists of :math:`M` non-interacting replicas 
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
a pair of temperatures/$\lambda$ vectors, we don't regard exchanging replicas as exchanging :math:`lambda`
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
we don't have their inverse to map the label $m$ back to the label :math:`i`.) That is, we have:

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
      &1 &, \text{if} \Delta \leq 0 \\
      \exp(&-\Delta) &, \text{if} \Delta >0
    \end{aligned}
  \end{cases}

Notably, if the systems of the replicas to be exchanged are sampling the same alchemical 
state (namely, :math:`m=n`) right before the exchange occurs, :math:`\Delta` will reduce to 
0, meaning the the exchange will always be accepted. 

MC schemes for swapping replicas
================================
- Same-state swapping

- Metropolis swapping 

- Equilibrated Metropolis swapping

Weight combination
==================
- Exponential averaging 

- Exponential averaging with histogram corrections


Transition matrix
=================
- Theoretical and experimental transition matrix 

- State transition matrix

- Replica transition matrix


Free energy calculation
=======================
