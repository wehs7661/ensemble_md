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
The :code:`analysis` module provides methods for performing data analysis for EEXE.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from ensemble_md.exceptions import ParseError
from ensemble_md.exceptions import ParameterError


def parse_transmtx(log_file, expanded_ensemble=True):
    """
    Parses the log file to get the transition matrix of an expanded ensemble
    or replica exchange simulation. Notably, a theoretical transition matrix
    is only available in expanded ensemble.

    Parameters
    ----------
    log_file : str
        The log file to be parsed.
    expanded_ensemble : bool
        Whether the simulation is expanded ensemble or replica exchange

    Returns
    -------
    empirical : np.array
        The final empirical state transition matrix.
    theoretical : None or np.array
        The final theoretical state transition matrix.
    diff_matrix : None or np.array
        The difference between the theortial and empirical state transition matrix (empirical - theoretical).
    """
    f = open(log_file, "r")
    lines = f.readlines()
    f.close()

    lines.reverse()

    n = -1
    theoretical_found, empirical_found = False, False
    for l in lines:  # noqa: E741
        n += 1
        if "Empirical Transition Matrix" in l:  # This will be found first
            empirical_found = True
            n_states = int(lines[n - 1].split()[-1])
            empirical = np.zeros([n_states, n_states])
            for i in range(n_states):
                if expanded_ensemble is True:
                    empirical[i] = [float(k) for k in lines[n - 2 - i].split()[:-1]]
                else:    # replica exchange
                    empirical[i] = [float(k) for k in lines[n - 2 - i].split()[1:-1]]

        if "Transition Matrix" in l and "Empirical" not in l:    # only occurs in expanded ensemble
            theoretical_found = True
            theoretical = np.zeros([n_states, n_states])
            for i in range(n_states):
                theoretical[i] = [float(k) for k in lines[n - 2 - i].split()[:-1]]

        if theoretical_found is True and empirical_found is True:
            diff_matrix = empirical - theoretical
            break

    if theoretical_found is False:
        theoretical = None
        diff_matrix = None

        if empirical_found is False:
            raise ParseError(f"No transition matrices found in {log_file}.")

    return empirical, theoretical, diff_matrix


def traj2transmtx(traj, N):
    """
    Compute the transition matrix given a trajectory. For example, if a state-space
    trajectory from a EXE or HREX simulation given, a state transition matrix is returned.
    If a trajectory showing transitions between replicas in a EEXE simulation is given,
    a replica transition matrix is returned.

    Parameters
    ---------
    traj : list
        A list of state indices showing the trajectory in the state space.
    N : int
        The size (N) of the expcted transition matrix (N by N).

    Returns
    -------
    transmtx : np.array
        The transition matrix computed from the trajectory
    """
    transmtx = np.zeros([N, N])
    for i in range(1, len(traj)):
        transmtx[traj[i - 1], traj[i]] += 1   # counts of transitions
    transmtx /= np.sum(transmtx, axis=1)[:, None]   # normalize the transition matrix
    transmtx[np.isnan(transmtx)] = 0   # for non-sampled state, there could be nan due to 0/0

    return transmtx


def calc_equil_prob(trans_mtx):
    """
    Calculates the equilibrium probability of each state from the state transition matrix.
    The input state transition matrix can be either left or right stochastic, although the
    left stochastic ones are not common in GROMACS. Generally, transition matrices in GROMACS
    are either doubly stochastic (replica exchange), or right stochastic (expanded ensemble).
    For the latter case, the staionary distribution vector is the left eigenvector corresponding
    to the eigenvalue 1 of the transition matrix. (For the former case, it's either left or
    right eigenvector corresponding to the eigenvalue 1 - as the left and right eigenvectors
    are the same for a doubly stochasti matrix.)

    Parameters
    ----------
    trans_mtx : np.array
        The input state transition matrix

    Returns
    -------
    equil_prob : np.array

    TODO
    ----
    Consider using PyEMMA instead.
    """
    if np.isclose(np.sum(trans_mtx[0]), 1):  # note that this also include doubly stochastic matrices
        # Right or doubly stachstic matrices - calculate the left eigenvector
        eig_vals, eig_vecs = np.linalg.eig(trans_mtx.T)
    elif np.isclose(np.sum(trans_mtx[:, 0]), 1):
        # Left stochastic matrix - calcualte the right eigenvector
        eig_vals, eig_vecs = np.linalg.eig(trans_mtx)
    else:
        raise ParseError("The input transition matrix is neither right nor left stochastic.")

    close_to_1_idx = np.isclose(eig_vals, 1, atol=1e-4)

    if np.sum(close_to_1_idx) == 0:
        print(f'Eigenvalues of the input transition matrix include: {eig_vals}')
        raise ParameterError('None of the eigenvalues are close to 1. Please check the input transition matrix.')

    # The eigenvector corresponding to the eigenvalue eig_vals[i] is eig_vecs[:, i]
    equil_prob = eig_vecs[:, close_to_1_idx]  # note that this is normalized
    equil_prob /= np.sum(equil_prob)   # So the sum of all components is 1

    return equil_prob


def calc_spectral_gap(trans_mtx):
    """
    Calculates the spectral gap of the input transition matrix.

    Parameters
    ----------
    trans_mtx : np.array
        The input state transition matrix

    Returns
    -------
    spectral_gap : float
        The spectral gap of the input transition matrix
    """
    if np.isclose(np.sum(trans_mtx[0]), 1):
        eig_vals, eig_vecs = np.linalg.eig(trans_mtx.T)
    elif np.isclose(np.sum(trans_mtx[:, 0]), 1):
        eig_vals, eig_vecs = np.linalg.eig(trans_mtx)
    else:
        raise ParseError("The input transition matrix is neither right nor left stochastic.")

    eig_vals = np.sort(eig_vals)[::-1]  # descending order
    if np.isclose(eig_vals[0], 1, atol=1e-4) is False:
        raise ParameterError(f'The largest eigenvalue of the input transition matrix {eig_vals[0]} is not close to 1.')

    spectral_gap = eig_vals[0] - eig_vals[1]

    return spectral_gap


def split_transmtx(trans_mtx, n_sim, n_sub):
    """
    Split the input transition matrix into blocks of smaller matrices corresponding to
    difrerent alchemical ranges of different replicas. Notably, the function assumes
    homogeneous shifts and number of states across replicas. Also, the blocks of the
    transition matrix is generally not doubly stochastic but right stochastic even if
    the input is doubly stochastic.

    Parameters
    ----------
    trans_mtx : np.array
        The input state transition matrix to split
    n_sim : int
        The number of replicas in EEXE.
    n_sub : int
        The number of states for each replica.

    Returns
    -------
    sub_mtx: A list of np.array
        Blocks of transition matrices split from the input.
    """
    sub_mtx = []
    ranges = [[i, i + n_sub] for i in range(n_sim)]   # A list of lists containing the min/max of alchemcial ranges
    for i in range(len(ranges)):
        bounds = ranges[i]
        # Below with numpy.copy, trans_mtx will be indirectly changed by the next line
        mtx = np.copy(trans_mtx[bounds[0]:bounds[1]][:, bounds[0]:bounds[1]])
        mtx /= np.sum(mtx, axis=1)[:, None]    # Divide each row by the sum of the row to rescale probabilities
        sub_mtx.append(mtx)

    return sub_mtx


def average_dg(g_vecs, frac, n_boots):
    """
    Average the differences between the weights of the coupled and uncoupled states.
    This can be an estimate of the free energy difference between two end states if
    :code:`g_vecs` is generated by :obj:`combine_weights`.

    Parameters
    ----------
    g_vecs : np.array
        An array of alchemical weights of the whole range of states.
    frac : float
        The average fraction.

    Returns
    -------
    dg_avg : float
        The averaged difference in the weights between the coupled and uncoupled states.
    dg_avg_err : float
        The error of :code:`dg_avg`.
    """
    N = len(g_vecs)
    dg = []
    for i in range(N):
        dg.append(g_vecs[i][-1] - g_vecs[i][0])
    n = int(np.floor(N * frac))
    dg_avg = np.mean(dg[-n:])
    dg_avg_err = np.std(dg_avg[-n:], ddof=1)

    return dg_avg, dg_avg_err


def plot_matrix(matrix, png_name, title=None, start_idx=0):
    """
    Visualizes a matrix a in heatmap.

    Parameters
    ----------
    matrix : np.array
        The matrix to be visualized
    png_name : str
        The file name of the output PNG file (including the extension).
    title : str
        The title of the plot
    start_idx : int
        The starting value of the state index
    """

    sns.set_context(
        rc={"family": "sans-serif",
            "sans-serif": ["DejaVu Sans"],
            "size": 5})

    K = len(matrix)
    plt.figure(figsize=(K / 1.5, K / 1.5))     # or figsize=(K / 1.5, K / 1.5)
    annot_matrix = np.zeros([K, K])  # matrix for annotating values

    mask = []
    for i in range(K):
        mask.append([])
        for j in range(len(matrix[0])):
            if matrix[i][j] < 0.005:
                mask[-1].append(True)
            else:
                mask[-1].append(False)

    for i in range(K):
        for j in range(K):
            annot_matrix[i, j] = round(matrix[i, j], 2)

    x_tick_labels = y_tick_labels = np.arange(start_idx, start_idx + K)
    ax = sns.heatmap(
        matrix,
        cmap="YlGnBu",
        linecolor="silver",
        linewidth=0.25,
        annot=annot_matrix,
        square=True,
        mask=matrix < 0.005,
        fmt=".2f",
        cbar=False,
        xticklabels=x_tick_labels,
        yticklabels=y_tick_labels,
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    cmap = cm.get_cmap("YlGnBu")  # to get the facecolor
    ax.set_facecolor(cmap(0))  # use the brightest color (value = 0)
    for _, spine in ax.spines.items():
        spine.set_visible(True)  # add frames to the heat map
    plt.annotate("$\lambda$", xy=(0, 0), xytext=(-0.45, -0.20))  # noqa: W605
    if title is None:
        pass
    else:
        plt.title(title, fontsize=10, weight="bold")
    plt.tight_layout(pad=1.0)

    plt.savefig(png_name, dpi=600)
    # plt.show()
    plt.close()
