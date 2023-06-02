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
The :obj:`.msm_analysis` module provides analysis methods based on Markov state models.
"""
import pyemma
import ruptures as rpt
import matplotlib.pyplot as plt

from ensemble_md.utils import utils


def plot_acf(models, n_tot, fig_name):
    """
    Plots the state index autocorrelation times for all configurations in a single plot

    Parameters
    ----------
    models : list
        A list of MSM models (built by PyEMMA) that have the :code:`correlation` method.
    n_tot : int
        The total number of states (whole range).
    fig_name : str
        The file name of the png file to be saved (with the extension).
    """
    plt.figure()
    for i in range(len(models)):
        if models[i] is not None:
            times, acf = models[i].correlation(models[i].active_set)
            plt.plot(times, acf, label=f'Configuration {i}')
    plt.xlabel('Time')   # Need to figure out what exactly this is ...
    plt.ylabel('Autocorrelation function (ACF)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)


def plot_its(trajs, lags, fig_name, dt=1, units='step'):
    """
    Plots the implied timescales as a function of lag time for all configurations
    in a subplot.

    Parameters
    ----------
    trajs : list
        A list of state-space trajectories.
    lags : list
        A list of lag times to examine.
    fig_name : str
        The file name of the png file to be saved (with the extension).
    dt : float
        Physical time between frames. The default is 1.
    units : str
        The units of dt. The default is 'ps'.

    Returns
    -------
    ts_list : list
        An list of instances of the :code:`ImpliedTimescales` class in PyEMMA.
    """
    ts_list = []
    n_rows, n_cols = utils.get_subplot_dimension(len(trajs))
    fig = plt.figure(figsize=(3 * n_cols, 2.5 * n_rows))
    for i in range(len(trajs)):
        # We convert trajs[i] to list to avoid BufferError: memoryview: underlying buffer is not C-contiguous
        ts = pyemma.msm.its(list(trajs[i]), lags=lags, show_progress=False)
        ts_list.append(ts)

        fig.add_subplot(n_rows, n_cols, i + 1)
        pyemma.plots.plot_implied_timescales(ts, dt=dt, units=units)
        plt.xlabel(f'Lag time ({units})')
        plt.ylabel(f'Implied timescale ({units})')
        plt.title(f'Configuration {i}', fontweight='bold')
        plt.grid()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)

    return ts_list


def decide_lagtimes(ts_list):
    """
    This function automatically estimates a lagtime for building an MSM for each configuration.
    Specifically, the lag time will be estimated by the change point detection enabled by
    ruptures for each (n-1) timescales (where n is the number of states). A good lag time
    should be long enough such that the timescale is roughly constant but short enough to be
    smaller than all timescales. If no lag time is smaller than all timescales, then a
    warning will be printed and a lag time of 1 will be returned in chosen_lags.

    Parameters
    ----------
    ts_list : list
        An list of instances of the ImpliedTimescales class in PyEMMA.

    Returns
    -------
    chosen_lags: list
        A list of lag time automatically determined for each configuration.
    """
    # Workflow: first find the timescales larger than the corressponding lag times,
    # then perform change change detection.
    chosen_lags = []
    print('     Suggested lag times (in trajectory frames) for each timescale curve of each configuration:')
    for i in range(len(ts_list)):   # for each configuration
        lag_list = []   # a list of lags chosen based on each timescale cure
        ts = ts_list[i]
        for j in range(len(ts.timescales[0])):  # compare each timescale curve with lag times
            ts_arr = ts.timescales[:, j]  # the j-th timescale curve, length: number of lagtimes
            ts_sub = ts_arr[ts_arr > ts.lagtimes]  # timescales that are larger than the corresponding lag times
            if len(ts_sub) <= 10:   # i.e. most timescales < lag time --> no appropirate lag time anyway, use 1
                lag_list.append(1)
            else:
                algo = rpt.Window(width=10, model='l2').fit(ts_sub)
                change_loc = algo.predict(n_bkps=1)  # this returns indices
                lag_list.append(ts.lagtimes[change_loc[0]])  # not sure if the first change point makes sense. Need to check.  # noqa: E501
        print(f'     - Configuration {i}: {lag_list}')  # noqa: E501

        # There might be cases like [6, 1, 1, 1] but using 6 is probably equally bad as 1.
        # If all are larger than one, using the max at least ensure all timescales are roughly constant.
        chosen_lags.append(max(lag_list))  # units: time frame

    return chosen_lags
