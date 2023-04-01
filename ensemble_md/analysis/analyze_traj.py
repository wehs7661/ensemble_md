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
The :obj:`.analyze_traj` module provides methods for analyzing trajectories in EEXE.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from alchemlyb.parsing.gmx import _get_headers as get_headers
from alchemlyb.parsing.gmx import _extract_dataframe as extract_dataframe
from ensemble_md.utils import utils


def extract_state_traj(dhdl):
    """
    Extracts the state-space trajectory from a GROMACS DHDL file.
    Note that the state indices here are local indices.

    Parameters
    ----------
    dhdl : str
        The filename of the GROMACS DHDL file to be parsed.

    Returns
    -------
    traj : list
        A list that represents that state-space trajectory
    """
    traj = list(extract_dataframe(dhdl, headers=get_headers(dhdl))['Thermodynamic state'])

    return traj


def stitch_trajs(files, rep_trajs, shifts=None, dhdl=True, col_idx=-1):
    """
    Stitches the state-space/CV-space trajectories for each configuration from DHDL files
    or PLUMED output files generated at different iterations.

    Parameters
    ----------
    files : list
        A list of lists of file names of GROMACS DHDL files or general GROMACS XVG files or PLUMED ouptput files.
        Specifically, :code:`files[i]` should be a list containing the files of interest from all iterations in
        replica :code:`i`.
    rep_trajs : list
        A list of lists that represents the replica space trajectories for each configuration. For example,
        :code:`rep_trajs[0] = [0, 2, 3, 0, 1, ...]` means that configuration 0 transitioned to replica 2, then
        3, 0, 1, in iterations 1, 2, 3, 4, ..., respectively.
    shifts : list
        A list of values for shifting the state indices for each replica. The length of the list
        should be equal to the number of replicas. This is only needed when :code:`dhdl=True`.
    dhdl : bool
        Whether the input files are GROMACS dhdl files, in which case trajectories of global alchemical indices
        will be generated. If :code:`dhdl=False`, the input files must be readable by `numpy.loadtxt` assuming that
        the start of a comment is indicated by either the :code:`#` or :code:`@` characters.
        Such files include any GROMACS XVG files or PLUMED output files (output by plumed driver, for instance).
        In this case, trajectories of the configurational collective variable of interest are generated.
        The default is :code:`True`.
    col_idx : int
        The index of the column to be extracted from the input files. This is only needed when :code:`dhdl=False`,
        By default, we extract the last column.

    Returns
    -------
    trajs : list
        A list that contains lists of state-space/CV-space trajectory (in global indices) for each configuration.
        For example, :code:`trajs[i]` is the state-space/CV-space trajectory of configuration :code:`i`.
    """
    n_configs = len(files)  # number of starting configurations
    n_iter = len(files[0])  # number of iterations per replica

    # First figure out which dhdl/plumed output files each configuration corresponds to
    # files_sorted[i] contains the dhdl/plumed output files for configuration i sorted based on iteration indices
    files_sorted = [[] for i in range(n_configs)]
    for i in range(n_configs):
        for j in range(n_iter):
            files_sorted[i].append(files[rep_trajs[i][j]][j])

    # Then, stitch the trajectories for each configuration
    trajs = [[] for i in range(n_configs)]  # for each configuration
    for i in range(n_configs):
        for j in range(n_iter):
            if j == 0:
                if dhdl:
                    traj = extract_state_traj(files_sorted[i][j])
                else:
                    traj = np.loadtxt(files_sorted[i][j], comments=['#', '@'])[:, -1]
            else:
                # Starting from the 2nd iteration, we get rid of the first time frame the first
                # frame of iteration n+1 the is the same as the last frame of iteration n
                if dhdl:
                    traj = extract_state_traj(files_sorted[i][j])[1:]
                else:
                    traj = np.loadtxt(files_sorted[i][j], comments=['#', '@'])[:, -1][1:]

            if dhdl:  # Trajectories of global alchemical indices will be generated.
                shift_idx = rep_trajs[i][j]
                traj = list(np.array(traj) + shifts[shift_idx])
            trajs[i].extend(traj)

    return trajs


def traj2transmtx(traj, N, normalize=True):
    """
    Computes the transition matrix given a trajectory. For example, if a state-space
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
    transmtx : numpy.ndarray
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
    Plots the time series of replicas visited by each configuration in a single plot.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the replica space trajectories of all configurations.
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : float or None, optional
        One trajectory timestep in ps. If None, it assumes there are no timeframes but MC steps.
    stride : int, optional
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


def plot_state_trajs(trajs, state_ranges, fig_name, dt=None, stride=1):
    """
    Plots the time series of states visited by each configuration in a subplot.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all configurations.
    state_ranges : list
        A list of lists of state indices. (Like the attribute :code:`state_ranges` in :code:`EnsemblEXE`.)
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : float or None, optional
        The time interval between consecutive frames of the trajectories. If None, it is assumed
        that the trajectories are in terms of Monte Carlo (MC) moves instead of timeframes, and
        the x-axis label is set to 'MC moves'. Default is None.
    stride : int, optional
        The stride for plotting the time series. The default is 10 if the length of
        any trajectory has more than 100,000 frames. Otherwise, it will be 1. Typically
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
        if len(trajs[0]) > 100000:
            stride = 10
        else:
            stride = 1

    # x_range = [-5, len(trajs[0]) - 1 + 5]
    x_range = [np.min(x), np.max(x)]
    y_range = [-0.2, np.max(trajs) + 0.2]
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

        if len(trajs[0]) > 100000:
            linewidth = 0.01
        else:
            linewidth = 1  # this is the default

        # Finally, plot the trajectories
        linewidth = 1  # this is the default
        plt.plot(x[::stride], trajs[i][::stride], color=colors[i], linewidth=linewidth)
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


def plot_state_hist(trajs, state_ranges, fig_name):
    """
    Plots the histograms of the state index for each configuration.

    Parameters
    ----------
    trajs : list
         A list of arrays that represent the state space trajectories of all configurations.
    state_ranges : list
        A list of lists of state indices. (Like the attribute :code:`state_ranges` in :obj:`.EnsembleEXE`.)
    fig_name : str
        The file name of the png file to be saved (with the extension).
    """
    n_configs = len(trajs)
    cmap = plt.cm.ocean  # other good options are CMRmap, gnuplot, terrain, turbo, brg, etc.
    colors = [cmap(i) for i in np.arange(n_configs) / n_configs]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    lower_bound = min(trajs[0]) - 0.5
    upper_bound = max(trajs[-1]) + 0.5
    for i in range(len(trajs)):
        plt.hist(trajs[i], np.arange(lower_bound, upper_bound + 1, 1), label=f'Configuration {i}', alpha=0.5, edgecolor='black', color=colors[i])  # noqa: E501
    plt.xticks(range(max(state_ranges[-1]) + 1))

    # Here we color the different regions to show alchemical ranges
    y_min, y_max = ax.get_ylim()
    for i in range(n_configs):
        bounds = [list(state_ranges[i])[0], list(state_ranges[i])[-1]]
        if i == 0:
            bounds[0] -= 0.5
        if i == n_configs - 1:
            bounds[1] += 0.5
        plt.fill_betweenx([y_min, y_max], x1=bounds[1] + 0.5, x2=bounds[0] - 0.5, color=colors[i], alpha=0.1, zorder=0)
    plt.xlim([lower_bound, upper_bound])
    plt.ylim([y_min, y_max])
    plt.xlabel('State index')
    plt.ylabel('Count')
    plt.grid()
    plt.legend()
    plt.savefig(f'{fig_name}', dpi=600)


def plot_transit_time(trajs, N, fig_prefix=None, dt=None, folder='.'):
    """
    Caclulcates and plots the average transit times for each configuration, including the time
    it takes from states 0 to k, from k to 0 and from 0 to k back to 0 (i.e. round-trip time).
    If there are more than 100 round-trips, 3 histograms corresponding to t_0k, t_k0 and t_roundtrip
    will be generated.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all configurations.
    N : int
        The total number of states in the whole alchemical range.
    fig_prefix : str
        A prefix to use for all generated figures.
    dt : float or None, optional
        One trajectory timestep in ps. If None, it assumes there are no timeframes but MC steps.
    folder : str, optional
        The directory for saving the figures

    Returns
    -------
    t_0k_list : list
        A list of transit time from states 0 to k for each configuration.
    t_k0_list : list
        A list of transit time from states k to 0 for each configuration.
    t_roundtrip_list : list
        A list of round-trip times for each configuration.
    units : str
        The units of the times.
    """
    if dt is None:
        x = np.arange(len(trajs[0]))
        units = 'step'
    else:
        x = np.arange(len(trajs[0])) * dt
        if max(x) >= 10000:
            x = x / 1000   # convert to ns
            units = 'ns'
        else:
            units = 'ps'

    # The lists below are for storing data corresponding to different configurations.
    t_0k_list, t_k0_list, t_roundtrip_list = [], [], []
    t_0k_avg, t_k0_avg, t_roundtrip_avg = [], [], []

    sci = False  # whether to use scientific notation in the y-axis in the plot
    for i in range(len(trajs)):
        traj = trajs[i]
        last_visited = None   # last visited end
        k = N - 1
        t_0, t_k = [], []   # time frames visting states 0 and k (k is the other end)

        # time spent from statkes 0 to k, k to 0 and the round-trip time (from 0 to k to 0)
        t_0k, t_k0, t_roundtrip = [], [], []

        end_0_found, end_k_found = None, None
        for t in range(len(traj)):
            if traj[t] == 0:
                end_0_found = True
                if last_visited != 0:
                    t_0.append(t)
                    if last_visited == k:
                        t_k0.append(t - t_k[-1])
                last_visited = 0
            if traj[t] == k:
                end_k_found = True
                if last_visited != k:
                    t_k.append(t)
                    if last_visited == 0:
                        t_0k.append(t - t_0[-1])
                last_visited = k

        # Here we figure out the round-trip time from t_0k and t_k0.
        if len(t_0k) != len(t_k0):   # then it must be len(t_0k) = len(t_k0) + 1 or len(t_k0) = len(t_0k) + 1, so we drop the last element of the larger list  # noqa: E501
            if len(t_0k) > len(t_k0):
                t_0k.pop()
            else:
                t_k0.pop()
        t_roundtrip = list(np.array(t_0k) + np.array(t_k0))

        if end_0_found is True and end_k_found is True:
            if dt is not None:
                units = 'ps'
                t_0k = list(np.array(t_0k) * dt)  # units: ps
                t_k0 = list(np.array(t_k0) * dt)  # units: ps
                t_roundtrip = list(np.array(t_roundtrip) * dt)  # units: ps
                if np.max([t_0k, t_k0, t_roundtrip]) >= 10000:
                    units = 'ns'
                    t_0k = list(np.array(t_0k) / 1000)   # units: ns
                    t_k0 = list(np.array(t_k0) / 1000)   # units: ns
                    t_roundtrip = list(np.array(t_roundtrip) / 1000)   # units: ns

            t_0k_list.append(t_0k)
            t_0k_avg.append(np.mean(t_0k))

            t_k0_list.append(t_k0)
            t_k0_avg.append(np.mean(t_k0))

            t_roundtrip_list.append(t_roundtrip)
            t_roundtrip_avg.append(np.mean(t_roundtrip))

            if sci is False and np.max([t_0k, t_k0, t_roundtrip]) >= 10000:
                sci = True
        else:
            t_0k_list.append([])
            t_k0_list.append([])
            t_roundtrip_list.append([])

    # Now we plot! (If there are no events, the figures will just be blank)
    meta_list = [t_0k_list, t_k0_list, t_roundtrip_list]
    y_labels = [
        f'Average transit time from states 0 to k ({units})',
        f'Average transit time from states k to 0 ({units})',
        f'Average round-trip time ({units})',
    ]
    fig_names = ['t_0k.png', 't_k0.png', 't_roundtrip.png']
    for t in range(len(meta_list)):
        t_list = meta_list[t]
        if all(not x for x in t_list):
            # If the nested list is empty, no plots will be generated.
            pass
        else:
            len_list = [len(i) for i in t_list]
            if np.max(len_list) <= 10:
                marker = 'o'
            else:
                marker = ''

            plt.figure()
            for i in range(len(t_list)):    # t_list[i] is the list for configuration i
                plt.plot(np.arange(len(t_list[i])) + 1, t_list[i], label=f'Configuration {i}', marker=marker)
            if np.array(t_list).shape != (1, 0):  # at least one configuration has at least one event
                if np.max(np.max((t_list))) >= 10000:
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.xlabel('Event index')
            plt.ylabel(f'{y_labels[t]}')
            plt.grid()
            plt.legend()
            if fig_prefix is None:
                plt.savefig(f'{folder}/{fig_names[t]}')
            else:
                plt.savefig(f'{folder}/{fig_prefix}_{fig_names[t]}', dpi=600)

            lens = [len(t_list[i]) for i in range(len(t_list))]
            if np.min(lens) >= 100:  # plot a histogram
                counts, bins = np.histogram(t_list[i])

                plt.figure()
                for i in range(len(t_list)):
                    plt.hist(t_list[i], bins=int(len(t_list[i]) / 20), label=f'Configuration {i}')
                    if max(counts) >= 10000:
                        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                plt.xlabel(f'{y_labels[t]}')
                plt.ylabel('Event count')
                plt.grid()
                plt.legend()
                if fig_prefix is None:
                    plt.savefig(f'{folder}/hist_{fig_names[t]}', dpi=600)
                else:
                    plt.savefig(f'{folder}/{fig_prefix}_hist_{fig_names[t]}', dpi=600)

    return t_0k_list, t_k0_list, t_roundtrip_list, units


def plot_g_vecs(f_g_vecs):
    pass
