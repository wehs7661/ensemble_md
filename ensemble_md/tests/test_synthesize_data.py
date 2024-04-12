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
Unit tests for the module synthesize_data.py.
"""
import pytest
import numpy as np
from ensemble_md.analysis import synthesize_data


def test_synthesize_traj():

    # Test 1: method == 'transmtx', check_row == N
    trans_mtx = np.array([[0.5, 0.5], [0.7, 0.3]])
    syn_traj = synthesize_data.synthesize_traj(trans_mtx, method='transmtx', start=1)
    assert syn_traj.shape[0] == 100000
    assert syn_traj[0] == 1

    # Test 2: method == 'transmtx', check_col == N
    trans_mtx = np.array([[0.5, 0.7], [0.5, 0.3]])
    syn_traj = synthesize_data.synthesize_traj(trans_mtx, n_frames=1000)
    assert syn_traj.shape[0] == 1000
    assert syn_traj[0] == 0

    # Test 3: method == 'transmtx', not normalized / invalid method
    trans_mtx = np.array([[0.2, 0.3], [0.5, 0.2]])
    with pytest.raises(ValueError, match='The input matrix is not normalized.'):
        synthesize_data.synthesize_traj(trans_mtx, method='transmtx')
    with pytest.raises(ValueError, match='Invalid method: test. The method must be either "transmtx" or "equil_prob".'):  # noqa: E501
        synthesize_data.synthesize_traj(trans_mtx, method='test')

    # Test 4: method == 'equil_prob' (Note that start should be ignored.)
    trans_mtx = np.array([[0.5, 0.5], [0.5, 0.5]])
    syn_traj = synthesize_data.synthesize_traj(trans_mtx, method='equil_prob', start=0)
    assert syn_traj.shape[0] == 100000

    # Test 5: Invalid value for start
    with pytest.raises(ValueError, match='The starting state 2 is out of the range of the input transition matrix.'):
        synthesize_data.synthesize_traj(trans_mtx, start=2)


def test_synthesize_transmtx():
    trans_mtx = np.array([[0.5, 0.5], [0.7, 0.3]])
    result = synthesize_data.synthesize_transmtx(trans_mtx, n_frames=1000000, seed=0)  # This should generate a diff_mtx close enough to a zero matrix.  # noqa: E501

    assert np.allclose(result[0], np.array([[0.5, 0.5], [0.7, 0.3]]), atol=0.01)
    assert len(result[1]) == 1000000
    assert np.allclose(result[2], np.array([[0, 0], [0, 0]]), atol=0.01)  # Basically the same as the first assertion
