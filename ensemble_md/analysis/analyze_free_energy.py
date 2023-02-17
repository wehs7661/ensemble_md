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
The :obj:`.analyze_free_energy` module provides functions for performing free energy calculations for EEXE simulations.
"""
import alchemlyb
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pymbar.timeseries import detect_equilibration, subsample_correlated_data  # noqa: E402
from alchemlyb.estimators import TI, BAR, MBAR  # noqa: E402
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk  # noqa: E402
from alchemlyb.preprocessing import subsampling  # noqa: E402
from ensemble_md.utils import utils  # noqa: E402
from ensemble_md.utils.exceptions import ParameterError  # noqa: E402


def preprocess_data(files, temp, spacing=1, get_u_nk=True, get_dHdl=False):
    """
    This function preprocesses :math:`dH/dλ` data obtained from the EEXE simulation.
    For each replica, it reads in :math:`dH/dλ` data from all iterations, concatenate
    them, remove the equilibrium region and and decorrelate the concatenated data. Notably,
    the data preprocessing protocol is basically the same as the one adopted in
    :code:`alchemlyb.subsampling.equilibrium_detection`.

    Parameters
    ----------
    files : list
        A list of naturally sorted dhdl files from all iterations of one replica.
    temp : float
        The simulation temperature in Kelvin.
    spacing : int
        The spacing (number of data points) to consider when subsampling the data.
    get_u_nk : bool
        Whether to get the u_nk data from the dhdl files. The default is True.
    get_dHdl : bool
        Whether to get the dHdl data from the dhdl files. the default is False.

    Returns
    -------
    preprocessed_u_nk : pd.Dataframe
        The preprocessed dHdl data that can serve as the input to free energy estimators.
    preprocessed_dHdl : pd.Dataframe
        The preprocessed dHdl data that can serve as the input to free energy estimators.
    """
    if get_u_nk is True:
        print('Collecting u_nk data from all iterations ...')
        u_nk = alchemlyb.concat([extract_u_nk(xvg, T=temp) for xvg in files])
        u_nk_series = subsampling.u_nk2series(u_nk)  # default method: 'dE'
        u_nk, u_nk_series = subsampling._prepare_input(u_nk, u_nk_series, drop_duplicates=True, sort=True)
        u_nk = subsampling.slicing(u_nk, step=spacing)
        u_nk_series = subsampling.slicing(u_nk_series, step=spacing)

        print('Subsampling and decorrelating the concatenated u_nk data ...')
        t, statinef, Neff_max = detect_equilibration(u_nk_series.values)

        print(f'  Adopted spacing: {spacing: .0f}')
        print(f' {t / len(u_nk_series) * 100: .1f}% of the u_nk data was in the equilibrium region and therfore discarded.')  # noqa: E501
        print(f'  Statistical inefficiency of u_nk: {statinef: .1f}')
        print(f'  Number of effective samples: {Neff_max: .0f}\n')

        u_nk_series_equil, u_nk_equil = u_nk_series[t:], u_nk[t:]
        indices = subsample_correlated_data(u_nk_series_equil, g=statinef)
        preprocessed_u_nk = u_nk_equil.iloc[indices]

    else:
        preprocessed_u_nk = None

    if get_dHdl is True:
        print('Collecting dHdl data from all iterations ...')
        dHdl = alchemlyb.concat([extract_dHdl(xvg, T=temp) for xvg in files])
        dHdl_series = subsampling.dhdl2series(dHdl)  # default method: 'dE'
        dHdl, dHdl_series = subsampling._prepare_input(dHdl, dHdl_series, drop_duplicates=True, sort=True)
        dHdl = subsampling.slicing(dHdl, step=spacing)
        dHdl_series = subsampling.slicing(dHdl_series, step=spacing)

        print('Subsampling and decorrelating the concatenated dHdl data ...')
        t, statinef, Neff_max = detect_equilibration(dHdl_series.values)

        print(f'  Adopted spacing: {spacing: .0f}')
        print(f' {t / len(dHdl_series) * 100: .1f}% of the dHdl data was in the equilibrium region and therfore discarded.')  # noqa: E501
        print(f'  Statistical inefficiency of dHdl: {statinef: .1f}')
        print(f'  Number of effective samples: {Neff_max: .0f}\n')

        dHdl_series_equil, dHdl_equil = dHdl_series[t:], dHdl[t:]
        indices = subsample_correlated_data(dHdl_series_equil, g=statinef)
        preprocessed_dHdl = dHdl_equil.iloc[indices]

    else:
        preprocessed_dHdl = None

    return preprocessed_u_nk, preprocessed_dHdl


def _calculate_df_adjacent(data, df_method="MBAR"):
    """
    An Internal function that generates a list of estimators fitting the input data
    and calculates at list of free energy between adjacent states for all replicas.

    Parameters
    ----------
    data : pd.Dataframe
        A list of dHdl or u_nk dataframes obtained from all replicas of the EEXE simulation of interest.
        Preferrably, the dHdl or u_nk data should be preprocessed by the function proprocess_data.

    Returns
    -------
    df_adjacent : list
        A list of free energy differences between adjacent states for all replicas.
    df_err_adjacent : list
        A list of uncertainties corresponding to the values of df_adjacent.
    """
    n_sim = len(data)
    estimators = []  # A list of objects of the corresponding class in alchemlyb.estimators
    for i in range(n_sim):
        if df_method == "TI":
            estimators.append(TI().fit(data[i]))
        elif df_method == "BAR":
            estimators.append(BAR().fit(data[i]))
        elif df_method == "MBAR":
            estimators.append(MBAR().fit(data[i]))
        else:
            raise ParameterError('Specified estimator not available.')

    df_adjacent = [list(np.array(estimators[i].delta_f_)[:-1, 1:].diagonal()) for i in range(n_sim)]
    df_err_adjacent = [list(np.array(estimators[i].d_delta_f_)[:-1, 1:].diagonal()) for i in range(n_sim)]

    return df_adjacent, df_err_adjacent


def _calculate_weighted_df(df_adjacent, df_err_adjacent, state_ranges, propagated_err=True):
    """
    An internal function that calculates a list of free energy differences between states i and i + 1.
    For free energy differences obtained from multiple replicas, an average weighted over all involved
    replicas is reported.

    Parameters
    ----------
    df_adjacent : list
        A list of free energy differences between adjacent states for all replicas.
    df_err_adjacent : list
        A list of uncertainties corresponding to the values of df_adjacent.
    state_ranges : list
        A list of lists of intergers that represents the alchemical states that can be sampled by different replicas.
    propagated_err : bool
        Whether to calculate the propagated error when taking the weighted averages for the free energy
        differences that can be obtained from multiple replicas. If False is specified, :code:`df_err`
        returned will be :code:`None`.

    Returns
    -------
    df : list
        A list of free energy differences between states i and i + 1.
    df_err : list
        A list of uncertainties of the free energy differences.
    overlap_bool : list
        overlap_bool[i] = True means that the i-th free energy difference (i.e. df[i]) was available
        in multiple replicas.
    """
    n_tot = state_ranges[-1][-1] + 1
    df, df_err, overlap_bool = [], [], []
    for i in range(n_tot - 1):
        # df_list is a list of free energy difference between sates i and i+1 in different replicas
        # df_err_list contains the uncertainties corresponding to the values of df_list
        df_list, df_err_list = [], []
        for j in range(len(state_ranges)):   # len(state_ranges) = n_sim
            if i in state_ranges[j] and i + 1 in state_ranges[j]:
                idx = state_ranges[j].index(i)
                df_list.append(df_adjacent[j][idx])
                df_err_list.append(df_err_adjacent[j][idx])
        overlap_bool.append(len(df_list) > 1)

        mean, error = utils.weighted_mean(df_list, df_err_list)
        df.append(mean)
        df_err.append(error)

    if propagated_err is False:
        df_err = None

    return df, df_err, overlap_bool


def calculate_free_energy(data, state_ranges, df_method="MBAR", err_method='propagate', n_bootstrap=None, seed=None):
    """
    Caculates the averaged free energy profile with the chosen method given dHdl or u_nk data obtained from
    all replicas of the EEXE simulation of interest. Available methods include TI, BAR, and MBAR. TI
    requires dHdl data while the other two require u_nk data.

    Parameters
    ----------
    data : pd.Dataframe
        A list of dHdl or u_nk dataframes obtained from all replicas of the EEXE simulation of interest.
        Preferrably, the dHdl or u_nk data should be preprocessed by the function proprocess_data.
    state_ranges : list
        A list of lists of intergers that represents the alchemical states that can be sampled by different replicas.
    df_method : str
        The method used to calculate the free energy profile. Available choices include "TI", "BAR", and "MBAR".
    err_method : str
        The method used to estimate the uncertainty of the free energy combined across multiple replicas.
        Available options include "propagate" and "bootstrap". The bootstrapping method is more accurate
        but much more computationally expensive than simple error propagation.
    n_bootstrap : int
        The number of bootstrap iterations. This parameter is used only when the boostrapping method is chosen to
        estimate the uncertainties of the free energies.
    seed : int
        The random seed for bootstrapping.

    Returns
    -------
    f : list
        The full-range free energy profile.
    f_err : list
        The uncertainty corresponding to the values in :code:`f`.
    estimators : list
        A list of estimators fitting the input data for all replicas. With this, the user
        can access all the free energies and their associated uncertainties for all states and replicas.
    """
    n_sim = len(data)
    n_tot = state_ranges[-1][-1] + 1
    df_adjacent, df_err_adjacent = _calculate_df_adjacent(data, df_method)
    df, df_err, overlap_bool = _calculate_weighted_df(df_adjacent, df_err_adjacent, state_ranges, propagated_err=True)

    if err_method == 'bootstrap':
        if seed is not None:
            print(f'Setting the random seed for boostrapping: {seed}')

        # Recalculate err with bootstrapping. (df is still the same and has been calculated above.)
        df_bootstrap = []
        sampled_data_all = [data[i].sample(n=len(data[i]) * n_bootstrap, replace=True, random_state=seed) for i in range(n_sim)]  # noqa: E501
        for b in range(n_bootstrap):
            sampled_data = [sampled_data_all[i].iloc[b * len(data[i]):(b + 1) * len(data[i])] for i in range(n_sim)]
            df_adjacent, df_err_adjacent = _calculate_df_adjacent(sampled_data, df_method)
            df_sampled, _, overlap_bool = _calculate_weighted_df(df_adjacent, df_err_adjacent, state_ranges, propagated_err=False)  # noqa: E501
            df_bootstrap.append(df_sampled)
        error_bootstrap = np.std(df_bootstrap, axis=0, ddof=1)

        # Replace the value in df_err with value in error_bootstrap if df_err corresponds to
        # the df between overlapping states
        for i in range(n_tot - 1):
            if overlap_bool[i] is True:
                print(f'Replaced the propagated error with the bootstrapped error for states {i} and {i + 1}: {df_err[i]:.5f} -> {error_bootstrap[i]:.5f}.')  # noqa: E501
                df_err[i] = error_bootstrap[i]

    df.insert(0, 0)
    df_err.insert(0, 0)
    f = [sum(df[:(i + 1)]) for i in range(len(df))]
    f_err = [np.sqrt(sum([x**2 for x in df_err[:(i+1)]])) for i in range(len(df_err))]

    return f, f_err


def plot_free_energy(f, f_err, fig_name):
    """
    Plot the free energy profile with error bars.

    Parameters
    ----------
    f : list
        The full-range free energy profile.
    f_err : list
        The uncertainty corresponding to the values in :code:`f`.
    fig_name : str
        The file name of the png file to be saved (with the extension).
    """
    plt.figure()
    plt.plot(range(len(f)), f, 'o-', c='#1f77b4')
    plt.errorbar(range(len(f)), f, yerr=f_err, fmt='o', capsize=2, c='#1f77b4')
    plt.xlabel('State')
    plt.ylabel('Free energy (kT)')
    plt.grid()
    plt.savefig(f'{fig_name}', dpi=600)


def average_weights(g_vecs, frac):
    """
    Average the differences between the weights of the coupled and uncoupled states.
    This can be an estimate of the free energy difference between two end states.

    Parameters
    ----------
    g_vecs : np.array
        An array of alchemical weights of the whole range of states as a function of
        simulation time, which is typically generated by :obj:`.combine_weights`.
    frac : float
        The fraction of g_vecs to average over. frac=0.2 means average the last 20% of
        the weight vectors.

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
