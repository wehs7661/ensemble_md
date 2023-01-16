####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################
import os
import math
import glob
import pymbar
import pickle
import natsort
import alchemlyb
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pymbar.timeseries import detect_equilibration, subsample_correlated_data
from alchemlyb.estimators import TI, BAR, MBAR
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
from alchemlyb.preprocessing import subsampling
from ensemble_md.utils import utils
from ensemble_md.utils.exceptions import ParameterError

def preprocess_data(files, temp, spacing=1, get_u_nk=True, get_dHdl=False):
    """
    This function preprocesses :math:`dH/d\lambda` data obtained from the EEXE simulation.
    For each replica, it reads in :math:`dH/d\lambda` data from all iterations, concatenate
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
    dHdl_data, u_nk_data = [], []

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
        print(f' {t / len(u_nk_series) * 100: .1f}% of the u_nk data was in the equilibrium region and therfore discarded.')
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
        print(f' {t / len(dHdl_series) * 100: .1f}% of the dHdl data was in the equilibrium region and therfore discarded.')
        print(f'  Statistical inefficiency of dHdl: {statinef: .1f}')
        print(f'  Number of effective samples: {Neff_max: .0f}\n')

        dHdl_series_equil, dHdl_equil = dHdl_series[t:], dHdl[t:]
        indices = subsample_correlated_data(dHdl_series_equil, g=statinef)
        preprocessed_dHdl = dHdl_equil.iloc[indices]

    else:
        preprocessed_dHdl = None

    return preprocessed_u_nk, preprocessed_dHdl

def gen_estimators(data, df_method="MBAR"):
    """
    Generate a list of estimators fitting the input data.

    Parameters
    ----------
    data : pd.Dataframe
        A list of dHdl or u_nk dataframes obtained from all replicas of the EEXE simulation of interest. 
        Preferrably, the dHdl or u_nk data should be preprocessed by the function proprocess_data. 

    Returns
    -------
    estimators : list
        A list of free energy estimators fitting the input data and with the free energy differences 
        (and their uncertanties) available. 
    """
    n_sim = len(data)
    estimators = []  # A list of objects of the corresponding class in alchemlyb.estimators
    for i in range(n_sim):
        if method == "TI": 
            estimators.append(TI().fit(data[i]))
        elif method == "BAR":
            estimators.append(BAR().fit(data[i]))
        elif method == "MBAR":
            estimators.append(MBAR().fit(data[i]))
        else:
            raise ParameterError('Specified estimator not available.')
    
    return estimators

def calculate_free_energy(data, state_ranges, df_method="MBAR", err_method='propagate', n_bootstrap=None):
    """
    Caculate the averaged free energy profile with the chosen method given dHdl or u_nk data obtained from all replicas of the 
    EEXE simulation of interest. Available methods include TI, BAR, and MBAR. TI requires dHdl data while the other two require
    u_nk data.

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
        The method used to estimate the uncertainty of the free energy combined across multiple replicas. Available options include "propagate" and "bootstrap". 
        The bootstrapping method is more accurate but much more computationally expensive than simple error propagation.
    n_bootstrap : int
        The number of bootstrap iterations. This parameter is used only when the boostrapping method is chosen to 
        estimate the uncertainties of the free energies.

    Returns
    -------
    estimators : list
        A list of estimators fitting the input data for all replicas.
    f : list
        The full-range free energy profile.
    f_err : list
        The uncertainty corresponding to the values in :code:`f`.

    
    df : list
        A full-range free energy profile with each entry averaged over replicas.
    err : list
        A list of uncertainties corresponding to the values in :code:`df`.
    df_all : list
        A list of lists of free energy profiles for each replica.
    err_all : list
        A list of lists of uncertainty corresponding to the values in :code:`df_all`.
    """
    n_sim = len(data)
    estimators = gen_estimators(data, df_method)

    n_tot = state_ranges[-1][-1] + 1
    df_all = [list(np.array(estimators[i].delta_f_)[:-1, 1:].diagonal()) for i in range(n_sim)]
    err_all = [list(np.array(estimators[i].d_delta_f_)[:-1, 1:].diagonal()) for i in range(n_sim)]
    
    df, err = [], []
    for i in range(n_tot - 1):
        df_list, err_list = [], []
        for j in range(n_sim):
            if i in state_ranges[j] and i + 1 in state_ranges[j]:
                idx = state_ranges[j].index(i)
                df_list.append(df_all[j][idx])
                err_list.append(err_all[j][idx])
        mean, error = utils.weighted_mean(df_list, err_list)
        df.append(mean)
        err.append(error)

    if err_method == 'bootstrap':
        # Recalculate err with bootstrapping. (df is still the same and has been calculated above.)
        err = []

    return df, err, df_all, err_all

def average_weights(g_vecs, frac):
    """
    Average the differences between the weights of the coupled and uncoupled states.
    This can be an estimate of the free energy difference between two end states.

    Parameters
    ----------
    g_vecs : np.array
        An array of alchemical weights of the whole range of states as a function of
        simulation time, which is typically generated by :code:`combine_weights`.
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
