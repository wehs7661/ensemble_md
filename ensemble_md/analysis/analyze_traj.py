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


def stitch_trajs(files, rep_trajs, shifts=None, dhdl=True, col_idx=-1, save=True):
    """
    Stitches the state-space/CV-space trajectories for each starting configuration from DHDL files
    or PLUMED output files generated at different iterations.

    Parameters
    ----------
    files : list
        A list of lists of file names of GROMACS DHDL files or general GROMACS XVG files or PLUMED ouptput files.
        Specifically, :code:`files[i]` should be a list containing the files of interest from all iterations in
        replica :code:`i`.
    rep_trajs : list
        A list of lists that represents the replica space trajectories for each starting configuration. For example,
        :code:`rep_trajs[0] = [0, 2, 3, 0, 1, ...]` means that starting configuration 0 transitioned to replica 2, then
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
    save : bool
        Whether to save the output trajectories as an NPY file.

    Returns
    -------
    trajs : list
        A list that contains lists of state-space/CV-space trajectory (in global indices) for each starting
        configuration. For example, :code:`trajs[i]` is the state-space/CV-space trajectory of starting
        configuration :code:`i`.
    """
    n_configs = len(files)  # number of starting configurations
    n_iter = len(files[0])  # number of iterations per replica

    # First figure out which dhdl/plumed output files each starting configuration corresponds to
    # files_sorted[i] contains the dhdl/plumed output files for starting configuration i sorted
    # based on iteration indices
    files_sorted = [[] for i in range(n_configs)]
    for i in range(n_configs):
        for j in range(n_iter):
            files_sorted[i].append(files[rep_trajs[i][j]][j])

    # Then, stitch the trajectories for each starting configuration
    trajs = [[] for i in range(n_configs)]  # for each starting configuration
    for i in range(n_configs):
        for j in range(n_iter):
            if j == 0:
                if dhdl:
                    traj = extract_state_traj(files_sorted[i][j])
                else:
                    traj = np.loadtxt(files_sorted[i][j], comments=['#', '@'])[:, col_idx]
            else:
                # Starting from the 2nd iteration, we get rid of the first time frame the first
                # frame of iteration n+1 the is the same as the last frame of iteration n
                if dhdl:
                    traj = extract_state_traj(files_sorted[i][j])[1:]
                else:
                    traj = np.loadtxt(files_sorted[i][j], comments=['#', '@'])[:, col_idx][1:]

            if dhdl:  # Trajectories of global alchemical indices will be generated.
                shift_idx = rep_trajs[i][j]
                traj = list(np.array(traj) + shifts[shift_idx])
            trajs[i].extend(traj)

    # Save the trajectories as an NPY file if desired
    if save is True:
        np.save('state_trajs.npy', trajs)

    return trajs


def stitch_trajs_for_sim(files, shifts=None, dhdl=True, col_idx=-1, save=True):
    """
    Stitches the state-space/CV-space trajectories in the same replica/simulation folder.
    That is, the output time series is contributed by multiple different trajectories (initiated by
    different starting configurations) to a certain alchemical range.

    Parameters
    ----------
    files : list
        A list of lists of file names of GROMACS DHDL files or general GROMACS XVG files
        or PLUMED output files. Specifically, :code:`files[i]` should be a list containing
        the files of interest from all iterations in replica :code:`i`.
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
    save : bool
        Whether to save the output trajectories as an NPY file.

    Returns
    -------
    trajs : list
        A list that contains lists of state-space/CV-space trajectory (in global indices) for each replica.
        For example, :code:`trajs[i]` is the state-space/CV-space trajectory of replica :code:`i`.
    """
    n_sim = len(files)      # number of replicas
    n_iter = len(files[0])  # number of iterations per replica
    trajs = [[] for i in range(n_sim)]
    for i in range(n_sim):
        for j in range(n_iter):
            if dhdl:
                traj = extract_state_traj(files[i][j])
            else:
                traj = np.loadtxt(files[i][j], comments=['#', '@'])[:, col_idx]

            if dhdl:
                traj = list(np.array(traj) + shifts[i])
            trajs[i].extend(traj)

    # Save the trajectories as an NPY file if desired
    if save is True:
        np.save('state_trajs_for_sim.npy', trajs)

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
    transmtx : np.ndarray
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
    Plots the time series of replicas visited by each trajectory in a single plot.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the all replica space trajectories.
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
            plt.plot(x[::stride], trajs[i][::stride], color=colors[i], label=f'Trajectory {i}')
        else:
            plt.plot(x[::stride], trajs[i][::stride], color=colors[i], label=f'Trajectory {i}', marker='o')

    if dt is None:
        plt.xlabel('MC moves')
    else:
        plt.xlabel(f'Time ({units})')

    plt.ylabel('Replica')
    plt.grid()
    plt.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{fig_name}', dpi=600)


def plot_state_trajs(trajs, state_ranges, fig_name, dt=None, stride=1, title_prefix='Trajectory'):
    """
    Plots the time series of states visited by each trajectory in a subplot.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all continuous trajectories.
    state_ranges : list
        A list of lists of state indices. (Like the attribute :code:`state_ranges` in :code:`EnsemblEXE`.)
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : float or None, optional
        The time interval between consecutive frames of the trajectories. If None, it is assumed
        that the trajectories are in terms of Monte Carlo (MC) moves instead of timeframes, and
        the x-axis label is set to 'MC moves'. Default is None.
    stride : int
        The stride for plotting the time series. The default is 10 if the length of
        any trajectory has more than 100,000 frames. Otherwise, it will be 1. Typically
        plotting more than 10 million frames can take a lot of memory.
    title_prefix : str
        The prefix shared by the titles of the subplots. For example, if :code:`title_prefix` is
        set to "Trajectory", then the titles of the subplots will be "Trajectory 0", "Trajectory 1", ..., etc.
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
        plt.title(f'{title_prefix} {i}', fontweight='bold')
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


def plot_state_hist(trajs, state_ranges, fig_name, stack=True, figsize=None, prefix='Trajectory', subplots=False, save_hist=True):  # noqa: E501
    """
    Plots the histograms of the state index for each trajectory.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all continuous trajectories.
    state_ranges : list
        A list of lists of state indices. (Like the attribute :code:`state_ranges` in :obj:`.EnsembleEXE`.)
    fig_name : str
        The file name of the png file to be saved (with the extension).
    stack : bool
        Whether to stack the histograms. Only meaningful when :code:`subplots` is :code:`False`.
    figsize : tuple
        A tuple specifying the length and width of the output figure. The
        default is :code:`(6.4, 4.8)` for cases having less than 30 states and :code:`(10, 4.8)` otherwise.
    prefix : str
        The prefix shared by the titles of the subplots, or the labels shown in the same plot.
        For example, if :code:`prefix` is set to "Trajectory", then the titles/labels of the
        will be "Trajectory 0", "Trajectory 1", ..., etc.
    subplots : bool
        Whether to plot the histogram in multiple subplots, with the title of
        each based on the value of :code:`prefix`.
    save_hist : bool
        Whether to save the histogram data.

    Returns
    -------
    hist_data : list
        The histogram data of the state index for each trajectory.
    """
    n_configs = len(trajs)
    n_states = max(max(state_ranges)) + 1
    cmap = plt.cm.ocean  # other good options are CMRmap, gnuplot, terrain, turbo, brg, etc.
    colors = [cmap(i) for i in np.arange(n_configs) / n_configs]

    hist_data = []
    lower_bound, upper_bound = -0.5, n_states - 0.5
    for traj in trajs:
        hist, bins = np.histogram(traj, bins=np.arange(lower_bound, upper_bound + 1, 1))
        hist_data.append(hist)
    if save_hist is True:
        np.save('hist_data.npy', hist_data)

    # Use the same bins for all histograms
    bins = bins[:-1]  # Remove the last bin edge because there are n+1 bin edges for n bins

    # Start plotting
    if figsize is None:
        if max(trajs[-1]) > 30:
            figsize = (10, 4.8)
        else:
            figsize = (6.4, 4.8)  # default

    fig = plt.figure(figsize=figsize)

    if subplots is False:
        # Initialize the list of bottom (only matters for stack = True)
        bottom = [0] * n_states

        ax = fig.add_subplot(111)
        y_max = 0
        for i in range(n_configs):
            max_count = np.max(bottom + hist_data[i])
            if max_count > y_max:
                y_max = max_count
            plt.bar(
                range(n_states),
                hist_data[i],
                align='center',
                width=1,
                color=colors[i],
                edgecolor='black',
                label=f'{prefix} {i}',
                alpha=0.5,
                bottom=bottom
            )

            if stack is True:
                bottom = [b + c for b, c in zip(bottom, hist_data[i])]

        plt.xticks(range(n_states))

        # Here we color the different regions to show alchemical ranges
        y_max *= 1.05
        for i in range(n_configs):
            bounds = [list(state_ranges[i])[0], list(state_ranges[i])[-1]]
            if i == 0:
                bounds[0] -= 0.5
            if i == n_configs - 1:
                bounds[1] += 0.5
            plt.fill_betweenx([0, y_max], x1=bounds[1] + 0.5, x2=bounds[0] - 0.5, color=colors[i], alpha=0.1, zorder=0)
        plt.xlim([lower_bound, upper_bound])
        plt.ylim([0, y_max])
        plt.xlabel('State index')
        plt.ylabel('Count')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{fig_name}', dpi=600)
    else:
        n_rows, n_cols = utils.get_subplot_dimension(n_configs)
        _, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows))
        for i in range(n_configs):
            plt.subplot(n_rows, n_cols, i + 1)
            plt.bar(
                state_ranges[i],
                hist_data[i][state_ranges[i]],
                align='center',
                width=1,
                edgecolor='black',
                alpha=0.5
            )

            plt.xticks(state_ranges[i], fontsize=8)
            plt.xlim([state_ranges[i][0] - 0.5, state_ranges[i][-1] + 0.5])
            plt.xlabel('State index')
            plt.ylabel('Count')
            plt.title(f'{prefix} {i}')
            plt.grid()
        plt.tight_layout()
        plt.savefig(f'{fig_name}', dpi=600)

    return hist_data


def plot_transit_time(trajs, N, fig_prefix=None, dt=None, folder='.'):
    """
    Caclulcates and plots the average transit times for each trajectory, including the time
    it takes from states 0 to k, from k to 0 and from 0 to k back to 0 (i.e. round-trip time).
    If there are more than 100 round-trips, 3 histograms corresponding to t_0k, t_k0 and t_roundtrip
    will be generated.

    Parameters
    ----------
    trajs : list
        A list of arrays that represent the state space trajectories of all continuous trajectories.
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
        A list of transit time from states 0 to k for each trajectory.
    t_k0_list : list
        A list of transit time from states k to 0 for each trajectory.
    t_roundtrip_list : list
        A list of round-trip times for each trajectory.
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

    # The lists below are for storing data corresponding to different trajectories.
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
            for i in range(len(t_list)):    # t_list[i] is the list for trajectory i
                plt.plot(np.arange(len(t_list[i])) + 1, t_list[i], label=f'Trajectory {i}', marker=marker)

            if max(max((t_list))) >= 10000:
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
                    plt.hist(t_list[i], bins=int(len(t_list[i]) / 20), label=f'Trajectory {i}')
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


def plot_g_vecs(g_vecs, refs=None, refs_err=None, plot_rmse=True):
    """
    Plots the alchemical weight for each alchemical intermediate state as a function of
    the iteration index. Note that the alchemical weight of the first state (which is always 0)
    is skipped. If the reference values are given, they will be plotted in the figure and
    an RMSE will be calculated.

    Parameters
    ----------
    g_vecs : np.array
        The alchemical weights of all states as a function of iteration index. The shape should
        be (n_iterations, n_states). Such an array can be directly read from :code:`g_vecs.npy`
        saved by :code:`run_EEXE`.
    refs : np.array
        The reference values of the alchemical weights.
    refs_err : list or np.array
        The errors of the reference values.
    plot_rmse : bool
        Whether to plot RMSE as a function of the iteration index.
    """
    # n_iter, n_state = g_vecs.shape[0], g_vecs.shape[1]
    g_vecs = np.transpose(g_vecs)
    n_sim = len(g_vecs)
    cmap = plt.cm.ocean  # other good options are CMRmap, gnuplot, terrain, turbo, brg, etc.
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]
    plt.figure()
    for i in range(1, len(g_vecs)):
        if len(g_vecs[0]) < 100:
            plt.plot(range(len(g_vecs[i])), g_vecs[i], label=f'State {i}', c=colors[i], linewidth=0.8, marker='o', markersize=2)  # noqa: E501
        else:  # plot without markers
            plt.plot(range(len(g_vecs[i])), g_vecs[i], label=f'State {i}', c=colors[i], linewidth=0.8)
    plt.xlabel('Iteration index')
    plt.ylabel('Alchemical weight (kT)')
    ax = plt.gca()
    x_range = ax.get_xlim()
    plt.xlim([0, x_range[1]])
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.2))

    if refs is not None:
        for i in range(1, len(refs)):
            plt.axhline(y=refs[i], c='black', linestyle='--', linewidth=0.5)
            if refs_err is not None:
                ax = plt.gca()
                x_range = ax.get_xlim()
                plt.fill_between(x_range, y1=refs[i] - refs_err[i], y2=refs[i] + refs_err[i], color='lightgreen')

        # Calculate the RMSE as a function of the iteration index
        RMSE_list = [np.sqrt(np.mean((g_vecs[:, i] - refs) ** 2)) for i in range(len(g_vecs[0]))]
        plt.text(0.02, 0.95, f'Final RMSE: {RMSE_list[-1]:.3f} kT', transform=ax.transAxes)
        print(f'Final RMSE: {RMSE_list[-1]: .3f} kT')

    plt.tight_layout()
    plt.savefig('g_vecs.png', dpi=600)

    if refs is not None and plot_rmse is True:
        plt.figure()
        plt.plot(range(len(g_vecs[i])), RMSE_list)
        plt.xlabel('Iteration index')
        plt.ylabel('RMSE in the alchemical weights (kT)')
        plt.grid()
        plt.savefig('g_vecs_rmse.png', dpi=600)


def get_swaps(EEXE_log='run_EEXE_log.txt'):
    """
    For each replica, identifies the states where exchanges were proposed and accepted.
    (Todo: We should be able to only use :code:`rep_trajs.npy` and :code:`state_trajs.npy`
    instead of parsing the EEXE log file to reach the same goal.)

    Parameters
    ----------
    EEXE_log : str
        The output log file of the EEXE simulation.

    Returns
    -------
    proposed_swaps : list
        A list of dictionaries showing where the swaps were proposed in
        each replica. Each dictionary (corresponding to one replica) have
        keys being the global state indices and values being the number of
        proposed swaps that occurred in the state indicated by the key.
    accepted_swaps : list
        A list of dictionaries showing where the swaps were accepted in
        each replica. Each dictionary (corresponding to one replica) have
        keys being the global state indices and values being the number of
        accepted swaps that occurred in the state indicated by the key.
    """
    f = open(EEXE_log, 'r')
    lines = f.readlines()
    f.close()

    state_list = []
    for line in lines:
        if 'Number of replicas: ' in line:
            n_sim = int(line.split('Number of replicas: ')[-1])
        if '- Replica' in line:
            state_list.append(eval(line.split('States ')[-1]))

        if 'Iteration' in line:
            break

    # Note that proposed_swaps and accepted_swaps are initialized in the same way
    proposed_swaps = [{i: 0 for i in state_list[j]} for j in range(n_sim)]  # Key: global state index; Value: The number of accepted swaps  # noqa: E501
    accepted_swaps = [{i: 0 for i in state_list[j]} for j in range(n_sim)]  # Key: global state index; Value: The number of accepted swaps  # noqa: E501
    state_trajs = [[] for i in range(n_sim)]  # the state-space trajectory for each REPLICA (not trajectory)
    for line in lines:
        if 'Simulation' in line and 'Global state' in line:
            rep = int(line.split(':')[0].split()[-1])
            state = int(line.split(',')[0].split()[-1])
            state_trajs[rep].append(state)

        if 'Proposed swap' in line:
            swap = eval(line.split(': ')[-1])
            proposed_swaps[swap[0]][state_trajs[swap[0]][-1]] += 1  # states_trajs[swap[0]][-1] is the last state sampled by swap[0]  # noqa: E501
            proposed_swaps[swap[1]][state_trajs[swap[1]][-1]] += 1  # states_trajs[swap[1]][-1] is the last state sampled by swap[1]  # noqa: E501

        if 'Swap accepted!' in line:
            accepted_swaps[swap[0]][state_trajs[swap[0]][-1]] += 1  # states_trajs[swap[0]][-1] is the last state sampled by swap[0]  # noqa: E501
            accepted_swaps[swap[1]][state_trajs[swap[1]][-1]] += 1  # states_trajs[swap[1]][-1] is the last state sampled by swap[1]  # noqa: E501

    return proposed_swaps, accepted_swaps


def plot_swaps(swaps, swap_type='', stack=True, figsize=None):
    """
    Plots the histogram of the proposed swaps or accepted swaps for each replica.

    Parameters
    ----------
    swaps : list
        A list of dictionaries showing showing the number of swaps for each
        state for each replica. This list could be either of the outputs from :obj:`.get_swaps`.
    swap_type : str
        The value should be either :code:`'accepted'` or :code:`'proposed'`. This value
        will only influence the name of y-axis and the output file name.
    stack : bool
        Whether to stack the histograms.
    figsize : tuple
        A tuple specifying the length and width of the output figure. The
        default is :code:`(6.4, 4.8)` for cases having less than 30 states and :code:`(10, 4.8)` otherwise.
    """
    n_sim = len(swaps)
    n_states = max(max(d.keys()) for d in swaps) + 1
    lower_bound, upper_bound = -0.5, n_states - 0.5
    state_ranges = [list(swaps[i].keys()) for i in range(n_sim)]
    cmap = plt.cm.ocean
    colors = [cmap(i) for i in np.arange(n_sim) / n_sim]

    # A new list of dictionaries, each of which consider all state indies
    full_data = [{state: d.get(state, 0) for state in range(n_states)} for d in swaps]  # d.get(state, 0) returns 0 if the state is unavilable  # noqa: E501

    # counts of swaps for all states
    counts_list = [[d[state] for state in range(n_states)] for d in full_data]

    # Initialize the list of bottom
    bottom = [0] * n_states

    if figsize is None:
        if n_states > 30:
            figsize = (10, 4.8)
        else:
            figsize = (6.4, 4.8)  # default

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    for i in range(n_sim):
        plt.bar(
            range(n_states),
            counts_list[i],
            align='center',
            width=1,
            color=colors[i],
            edgecolor='black',
            label=f'Replica {i}',
            alpha=0.5,
            bottom=bottom
        )

        if stack is True:
            bottom = [b + c for b, c in zip(bottom, counts_list[i])]

    plt.xticks(range(n_states))

    # Here we color the different regions to show alchemical ranges
    y_min, y_max = ax.get_ylim()
    for i in range(n_sim):
        bounds = [list(state_ranges[i])[0], list(state_ranges[i])[-1]]
        if i == 0:
            bounds[0] -= 0.5
        if i == n_sim - 1:
            bounds[1] += 0.5
        plt.fill_betweenx([y_min, y_max], x1=bounds[1] + 0.5, x2=bounds[0] - 0.5, color=colors[i], alpha=0.1, zorder=0)
    plt.xlim([lower_bound, upper_bound])
    # plt.ylim([y_min, y_max])
    plt.xlabel('State')
    plt.ylabel(f'Number of {swap_type} swaps')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if swap_type == '':
        plt.savefig('swaps.png', dpi=600)
    else:
        plt.savefig(f'{swap_type}_swaps.png', dpi=600)
