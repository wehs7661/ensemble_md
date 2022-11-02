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
The :code:`msm_analysis` module provides analysis methods based on Markov state models.
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
            times, acf = models[i].correlation(range(n_tot))
            plt.plot(times, acf, label=f'Configuration {i}')
    plt.xlabel('Time')   # Need to figure out what exactly this is ...
    plt.ylabel('Autocorrelation function (ACF)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)

def plot_its(trajs, lags, fig_name, dt=1, units='step'):
    """
    Plot the implied timescales as a function of lag time for all configurations
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
    chosen_lags: list
        A list of lag time automatically determined for each configuration.
    """
    n_rows, n_cols = utils.get_subplot_dimension(len(trajs))
    fig = plt.figure(figsize=(2.5 * n_cols, 2.5 * n_rows))
    for i in range(len(trajs)):
        ts = pyemma.msm.its(trajs[i], lags=lags, show_progress=False)

        fig.add_subplot(n_rows, n_cols, i + 1)
        pyemma.plots.plot_implied_timescales(ts, dt=dt, units=units)
        plt.xlabel(f'Lag time ({units})')
        plt.ylabel(f'Implied timescale ({units})')
        plt.title(f'Configuration {i}', fontweight='bold')
        plt.grid()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=600)

