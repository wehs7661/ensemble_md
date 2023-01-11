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

from pymbar.timeseries import statisticalInefficiency
from alchemlyb.estimators import TI, BAR, AutoMBAR
from alchemlyb.parsing.gmx import extract_dHdl, extract_u_nk
from alchemlyb.preprocessing import equilibrium_detection
from ensemble_md.utils import utils
from ensemble_md.utils.exceptions import ParameterError

def preprocess_data(files, temp, spacing=1, get_u_nk=True, get_dHdl=False):
    """
    This function preprocesses :math:`dH/d\lambda` data obtained from the EEXE simulation.
    For each replica, it reads in :math:`dH/d\lambda` data from all iterations, concatenate
    them, and decorrelate the concatenated data. 
    
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
        u_nk = u_nk.loc[~u_nk.index.duplicated(keep='last')]
        
        print('Subsampling and decorrelating the concatenated u_nk data ...')
        truncated_u_nk = equilibrium_detection(u_nk, u_nk.iloc[:, 0], step=spacing)
        g_u_nk = statisticalInefficiency(u_nk.iloc[:, 0])
        preprocessed_u_nk = truncated_u_nk[::math.ceil(g_u_nk)]
        print(f'  Statistical inefficiency of u_nk: {g_u_nk:.3f} (adjusted to {math.ceil(g_u_nk)}) ==> {len(preprocessed_u_nk)} effective samples.')

        with open('u_nk_data.pickle', 'wb') as handle:
            pickle.dump(preprocessed_u_nk, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        preprocessed_u_nk = None

    if get_dHdl is True:
        print('Collecting dHdl data from all iterations ...')
        dHdl = alchemlyb.concat([extract_dHdl(xvg, T=temp) for xvg in files])
        dHdl = dHdl.loc[~dHdl.index.duplicated(keep='last')]
        
        print('Subsampling and decorrelating the concatenated dHdl data ...')
        truncated_dHdl = equilibrium_detection(dHdl, dHdl.iloc[:, 0], step=spacing)
        g_dHdl = statisticalInefficiency(dHdl.iloc[:, 0])
        preprocessed_dHdl = truncated_dHdl[::math.ceil(g_dHdl)]
        print(f'  Statistical inefficiency of dHdl: {g_dHdl:.3f} (adjusted to {math.ceil(g_dHdl)}) ==> {len(preprocessed_dHdl)} effective samples.\n')

        with open('dHdl_data.pickle', 'wb') as handle:
            pickle.dump(preprocessed_dHdl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        preprocessed_dHdl = None

    return preprocessed_u_nk, preprocessed_dHdl

def calculate_free_energy(data, state_ranges, method="MBAR"):
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
    method : str
        The method used to calculate the free energy profile. Available choices include "TI", "BAR", and "MBAR".

    Returns
    -------
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
    estimators = []  # A list of objects of the corresponding class in alchemlyb.estimators
    for i in range(n_sim):
        if method == "TI": 
            estimators.append(TI().fit(data[i]))
        elif method == "BAR":
            estimators.append(BAR().fit(data[i]))
        elif method == "MBAR":
            estimators.append(AutoMBAR().fit(data[i]))
        else:
            raise ParameterError('Specified estimator not available.')

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
