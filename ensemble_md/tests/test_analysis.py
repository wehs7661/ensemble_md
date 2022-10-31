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
Unit tests for the module analysis.py.
"""
import os
import pytest
import numpy as np
from ensemble_md.analysis import analyze_matrix
from ensemble_md.utils.exceptions import ParseError

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def test_parse_transmtx():
    A1, B1, C1 = analyze_matrix.parse_transmtx(os.path.join(input_path, 'EXE.log'))
    A1_expected = np.array([[0.5, 0.34782609, 0.15000001, 0, 0, 0],
       [0.34782609, 0.18181819, 0.15789473, 0.17647059, 0.10526316, 0.125     ],   # noqa: E128, E202, E203
       [0.15000001, 0.15789473, 0.        , 0.14285715, 0.4375    , 0.07692308],   # noqa: E202, E203
       [0.        , 0.17647059, 0.14285715, 0.16666667, 0.2857143 , 0.18181819],   # noqa: E202, E203
       [0.        , 0.10526316, 0.4375    , 0.2857143 , 0.        , 0.23076923],   # noqa: E202, E203
       [0.        , 0.125     , 0.07692308, 0.18181819, 0.23076923, 0.2       ]])  # noqa: E202, E203

    B1_expected = np.array([[0.5035941 , 0.27816516, 0.15014392, 0.06404528, 0.01056806, 0.00325544],  # noqa: E128, E202, E203, E501
       [0.27816516, 0.1430219 , 0.22873016, 0.12587893, 0.0782651 , 0.07551135],   # noqa: E128, E202, E203
       [0.15014392, 0.22873016, 0.12687601, 0.22781774, 0.19645289, 0.14166102],   # noqa: E202, E203
       [0.06404528, 0.12587893, 0.22781774, 0.0986531 , 0.27133784, 0.18215007],   # noqa: E202, E203
       [0.01056806, 0.0782651 , 0.19645289, 0.27133784, 0.06328789, 0.36931726],   # noqa: E202, E203
       [0.00325544, 0.07551135, 0.14166102, 0.18215007, 0.36931726, 0.2982755 ]])  # noqa: E202, E203

    C1_expected = np.array([[-3.5941000e-03,  6.9660930e-02, -1.4391000e-04, -6.4045280e-02, -1.0568060e-02, -3.2554400e-03],  # noqa: E501, E128
       [ 6.9660930e-02,  3.8796290e-02, -7.0835430e-02,  5.0591660e-02, 2.6998060e-02,  4.9488650e-02],  # noqa: E128, E201, E501
       [-1.4391000e-04, -7.0835430e-02, -1.2687601e-01, -8.4960590e-02, 2.4104711e-01, -6.4737940e-02],
       [-6.4045280e-02,  5.0591660e-02, -8.4960590e-02,  6.8013570e-02, 1.4376460e-02, -3.3188000e-04],
       [-1.0568060e-02,  2.6998060e-02,  2.4104711e-01,  1.4376460e-02, -6.3287890e-02, -1.3854803e-01],
       [-3.2554400e-03,  4.9488650e-02, -6.4737940e-02, -3.3188000e-04, -1.3854803e-01, -9.8275500e-02]])

    np.testing.assert_array_almost_equal(A1, A1_expected)
    np.testing.assert_array_almost_equal(B1, B1_expected)
    np.testing.assert_array_almost_equal(C1, C1_expected)

    log = os.path.join(input_path, 'EXE_0.log')
    with pytest.raises(ParseError, match=f'No transition matrices found in {log}.'):
        A2, B2, C2 = analyze_matrix.parse_transmtx(log)


def test_plot_matrix():
    """
    We can only check if the figures are generated. Not really able to check how they look like.
    """
    A1, B1, C1 = analyze_matrix.parse_transmtx(os.path.join(input_path, 'EXE.log'))
    analyze_matrix.plot_matrix(A1, 'test_1.png')
    analyze_matrix.plot_matrix(B1, 'test_2.png')
    analyze_matrix.plot_matrix(C1, 'test_3.png')

    assert os.path.exists('test_1.png') is True
    assert os.path.exists('test_2.png') is True
    assert os.path.exists('test_3.png') is True

    os.remove('test_1.png')
    os.remove('test_2.png')
    os.remove('test_3.png')
