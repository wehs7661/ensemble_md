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
The :code:`analysis_trajs` module provides methods for analyzing trajectories in EEXE.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from alchemlyb.parsing.gmx import _get_headers as get_headers
from alchemlyb.parsing.gmx import _extract_dataframe as extract_dataframe
from ensemble_md.utils import utils


def extract_state_traj(dhdl):
    """
    Extract the state-space trajectory from a dhdl file.

    Parameters
    ----------
    dhdl : str
        The filename of the dhdl file to be parsed.

    Returns
    -------
    traj : list
        A list that represents that state-space trajectory
    """
    traj = list(extract_dataframe(dhdl, headers=get_headers(dhdl))['Thermodynamic state'])

    return traj


def stitch_trajs(dhdl_files, rep_trajs, shifts):
    """
    Stitch the state-space trajectories for each configuration from dhdl files of different iterations.

    Parameters
    ----------
    dhdl_files : list
        A list of list of dhdl file names. Specifically, dhdl_files[i] should be a list containing
        the filenames of the dhdl files from all iterations in replica i.
    rep_trajs : list
        A list of list that represents the replica space trajectories for each configuration. For example,
        rep_trajs[0] = [0, 2, 3, 0, 1, ...] means that configuration 0 transitioned to replica 2, then
        3, 0, 1, in iterations 1, 2, 3, 4, ..., respectively.
    shifts : list
        A list of values for shifting the state indices for each replica. The length of the list
        should be equal to the number of replicas.


    Returns
    -------
    state_trajs : list
        A list that contains lists of state-space trajectory for each configuration. For example,
        state_trajs[i] is the state-space trajectory of configuration i.
    """
    n_configs = len(dhdl_files)
    n_iter = len(dhdl_files[0])

    # First figure out which dhdl files each configuration corresponds to
    # dhdl_sorted[i] contains the dhdl files for configuration i sorted based on iteration indices
    dhdl_sorted = [[] for i in range(n_configs)]
    for i in range(n_configs):
        for j in range(n_iter):
            dhdl_sorted[i].append(dhdl_files[rep_trajs[i][j]][j])

    # Then, stitch the trajectories for each configuration
    state_trajs = [[] for i in range(n_configs)]  # for each configuration
    for i in range(n_configs):
        for j in range(n_iter):
            if j == 0:
                traj = extract_state_traj(dhdl_sorted[i][j])
            else:
                # Get rid of the first time frame starting from the 2nd iteration because the first
                # frame of iteration n+1 the is the same as the last frame of iteration n
                traj = extract_state_traj(dhdl_sorted[i][j])[1:]
            shift_idx = rep_trajs[i][j]
            traj = list(np.array(traj) + shifts[shift_idx])
            state_trajs[i].extend(traj)

    return state_trajs


def traj2transmtx(traj, N, normalize=True):
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
    normalize : bool
        Whether to normalize the matrix so that each row sum to 1. If False, then
        the entries will be the counts of transitions.

    Returns
    -------
    transmtx : np.array
        The transition matrix computed from the trajectory
    """
    transmtx = np.zeros([N, N])
    for i in range(1, len(traj)):
        transmtx[traj[i - 1], traj[i]] += 1   # counts of transitions
    if normalize is True:
        transmtx /= np.sum(transmtx, axis=1)[:, None]   # normalize the transition matrix
        transmtx[np.isnan(transmtx)] = 0   # for non-sampled state, there could be nan due to 0/0

    return transmtx


def plot_rep_trajs(trajs, fig_name, dt=None, stride=None):
    """
    Plot the time series of replicas visited by each configuration in a single plot.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the replica space trajectories of all configurations.
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : str or float
        The timestep between frames in a trajectory in ps. If None, it assumes there are no timeframes but MC steps.
    stride : int
        The stride for plotting the time series. The default is 100 if the length of
        any trajectory has more than one million frames. Otherwise, it will be 1. Typically
        plotting more than 10 million frames can take a lot of memory.
    """
    n_sim = len(trajs)
    cmap = plt.cm.ocean  # other good options are CMRmap, gnuplot, terrain, turbo, brg, etc.
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    if dt is None:
        x = np.arange(len(trajs[0]))
    else:
        x = np.arange(len(trajs[0])) * dt
        if max(x) >= 10000:
            x = x / 1000   # convert to ns
            units = 'ns'
        else:
            units = 'ps'

    if stride is None:
        if len(trajs[0]) > 1000000:
            stride = 100
        else:
            stride = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(n_sim):
        if len(trajs[0]) >= 100:  # don't show the markers
            plt.plot(x[::stride], trajs[i][::stride], color=colors[i], label=f'Configuration {i}')
        else:
            plt.plot(x[::stride], trajs[i][::stride], color=colors[i], label=f'Configuration {i}', marker='o')

    if dt is None:
        plt.xlabel('MC moves')
    else:
        plt.xlabel(f'Time ({units})')

    plt.ylabel('Replica')
    plt.grid()
    plt.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{fig_name}', dpi=600)


def plot_state_trajs(trajs, state_ranges, fig_name, dt=None, stride=None):
    """
    Plot the time series of states visited by each configuration in a subplot.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all configurations.
    state_ranges : list
        A list of sets of state indices. (Like the attribute :code:`state_ranges` in :code:`EnsemblEXE`.)
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : str or float
        The timestep between frames in a trajectory in ps. If None, it assumes there are no timeframes but MC steps.
    stride : int
        The stride for plotting the time series. The default is 100 if the length of
        any trajectory has more than one million frames. Otherwise, it will be 1. Typically
        plotting more than 10 million frames can take a lot of memory.
    """
    n_sim = len(trajs)
    cmap = plt.cm.ocean  # other good options are CMRmap, gnuplot, terrain, turbo, brg, etc.
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    if dt is None:
        x = np.arange(len(trajs[0]))
    else:
        x = np.arange(len(trajs[0])) * dt
        if max(x) >= 10000:
            x = x / 1000   # convert to ns
            units = 'ns'
        else:
            units = 'ps'

    if stride is None:
        if len(trajs[0]) > 1000000:
            stride = 100
        else:
            stride = 1

    # x_range = [-5, len(trajs[0]) - 1 + 5]
    x_range = [np.min(x), np.max(x)]
    y_range = [-0.2, np.max(trajs) - 1 + 0.2]
    n_configs = len(trajs)
    n_rows, n_cols = utils.get_subplot_dimension(n_configs)
    _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    for i in range(n_configs):
        plt.subplot(n_rows, n_cols, i + 1)

        # First color different regions showing alchemical ranges
        for j in range(n_configs):
            bounds = [list(state_ranges[j])[0], list(state_ranges[j])[-1]]
            if j == 0:
                bounds[0] -= 0.5
            if j == n_configs - 1:
                bounds[1] += 0.5
            plt.fill_between(x_range, y1=bounds[1], y2=bounds[0], color=colors[j], alpha=0.1)

        # Then plot the trajectories
        plt.plot(x[::stride], trajs[i][::stride], color=colors[i])
        if dt is None:
            plt.xlabel('MC moves')
        else:
            plt.xlabel(f'Time ({units})')

        plt.ylabel('State')
        plt.title(f'Configuration {i}', fontweight='bold')
        if len(trajs[0]) >= 10000:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.grid()

    # Remove redundant subplots
    n_rm = n_cols * n_rows - n_configs
    for i in range(n_rm):
        ax.flat[-1 * (i + 1)].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{fig_name}', dpi=600)


def plot_transit_time(trajs, N, fig_name, dt=None):
    """
    Caclulcates and plots the average end-to-end transit time for each configuraiton.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all configurations.
    N : int
        The total number of states in the whole alchemical range.
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : str or float
        The timestep between frames in a trajectory in ps. If None, it assumes there are no timeframes but MC steps.

    Returns
    -------
    t_transit_list : list
        A list of end-to-end transit time for each configuration.
    units : str
        The units of the end-to-end-transit time.
    """
    if dt is None:
        x = np.arange(len(trajs[0]))
    else:
        x = np.arange(len(trajs[0])) * dt
        if max(x) >= 10000:
            x = x / 1000   # convert to ns
            units = 'ns'
        else:
            units = 'ps'

    t_transit_list = []
    t_transit_avg = []
    plt.figure()
    sci = False  # whether to use scientific notation in the y-axis in the plot
    for i in range(len(trajs)):
        traj = trajs[i]
        last_visited = None
        k = N - 1
        t_0, t_k = [], []   # time frames visting states 0 and k (k is the other end)
        t_transit = []  # end-to-end transit time
        end_0_found, end_k_found = None, None

        for t in range(len(traj)):
            if traj[t] == 0:
                end_0_found = True
                if last_visited != 0:
                    t_0.append(t)
                    if last_visited == k:
                        t_transit.append(t - t_k[-1])
                last_visited = 0
            if traj[t] == k:
                end_k_found = True
                if last_visited != k:
                    t_k.append(t)
                    if last_visited == 0:
                        t_transit.append(t - t_0[-1])
                last_visited = k

        if end_0_found is True and end_k_found is True:
            if dt is None:
                units = 'step'
            else:
                t_transit = list(np.array(t_transit) * dt)  # units: ps
                units = 'ps'
                if np.max(t_transit) >= 10000:
                    units = 'ns'
                    t_transit = list(np.array(t_transit) / 1000)   # units: ns

            if np.max(t_transit) >= 10000:
                sci = True
            t_transit_list.append(t_transit)
            t_transit_avg.append(np.mean(t_transit))
            plt.plot(np.arange(len(t_transit)) + 1, t_transit, label=f'Configuration {i}')
        else:
            t_transit_list.append(None)

    if sci:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xlabel('Event index')
    plt.ylabel(f'Averaged end-to-end transit time ({units})')
    plt.grid()
    plt.legend()
    plt.savefig(fig_name, dpi=600)

    return t_transit_list, units
