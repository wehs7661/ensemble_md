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
Unit tests for the module clustering.py.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from ensemble_md.analysis import clustering


def test_cluster_traj():
    pass


def test_get_cluster_info():
    cluster_log = 'ensemble_md/tests/data/cluster.log'
    results = cluster_info = clustering.get_cluster_info(cluster_log)

    assert results[0] == [0.0236461, 0.316756]
    assert results[1] == 0.182848
    assert results[2] == 2


def test_get_cluster_members():
    cluster_log = 'ensemble_md/tests/data/cluster.log'
    clusters, sizes = clustering.get_cluster_members(cluster_log)
    assert clusters == {
        1: [0, 178, 184, 186, 300, 302, 304, 306, 308, 310, 312, 318, 362, 366, 370, 372, 374, 376, 378, 380, 382, 390, 460, 464, 468, 470, 476],  # noqa: E501
        2: [11910, 11992, 11996, 12014, 12054, 12058, 12062, 12064, 12084, 12092, 12098, 12100, 12102, 12104, 12106, 12108, 12110, 12112, 12114, 12116, 12118, 12120, 12242, 12262, 12310, 12318, 12330, 12334, 12340],  # noqa: E501
    }
    assert sizes == {1: 27/56, 2: 29/56}


@patch('ensemble_md.analysis.clustering.plt')
def test_analyze_transitions(mock_plt):
    # Test 1: No transitions
    clusters = {
        1: [0, 178, 184, 186, 300, 302, 304, 306, 308, 310, 312, 318, 362, 366, 370, 372, 374, 376, 378, 380, 382, 390, 460, 464, 468, 470, 476],  # noqa: E501
    }
    results = clustering.analyze_transitions(clusters)
    assert results[0] == np.array([[1]])
    assert (results[1] == np.ones(len(clusters[1]))).all
    assert results[2] == {}

    # Test 2: More transitions
    clusters = {
        1: [0, 2, 4, 6, 20, 22, 24, 34, 36, 38, 62, 64, 66, 80, 82, 84, 86],
        2: [8, 10, 12, 26, 28, 30, 32, 48, 50, 52, 54, 74, 76, 78, 88, 90, 92, 94, 96],
        3: [14, 16, 18, 40, 42, 44, 46, 56, 58, 60, 68, 70, 72],
    }
    results = clustering.analyze_transitions(clusters, normalize=False)
    traj = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 1, 1, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2]  # noqa: E501
    assert np.sum(results[0]) == sum([len(clusters[i]) for i in clusters]) - 1
    assert (results[0] == np.array([[12, 3, 2], [2, 14, 2], [2, 2, 9]])).all()
    assert (results[1] == traj).all()
    assert results[2] == {
        (1, 2): [8, 26, 34, 80, 88],
        (2, 3): [14, 48, 56, 74],
        (1, 3): [20, 40, 62, 68],
    }

    # Test 3: Test the plotting functionalities
    # 3-1. plt_type = 'bar'
    mock_ax, mock_fig, mock_obj = MagicMock(), MagicMock(), MagicMock()
    mock_plt.figure.return_value = mock_fig
    mock_fig.add_subplot.return_value = mock_ax
    mock_plt.MaxNLocator.return_value = mock_obj

    results = clustering.analyze_transitions(clusters, plot_type='bar')
    
    mock_plt.figure.called_once()
    mock_fig.add_subplot.assert_called_once_with(111)
    mock_plt.bar.called_once()
    mock_plt.xlabel.assert_called_once_with('Cluster index')
    mock_plt.ylabel.assert_called_once_with('Number of configurations')
    mock_plt.grid.assert_called_once()
    mock_ax.xaxis.set_major_locator.assert_called_once_with(mock_obj)
    mock_ax.yaxis.set_major_locator.assert_called_once_with(mock_obj)
    mock_plt.savefig.assert_called_once_with('cluster_distribution.png', dpi=600)

    assert list(mock_plt.bar.call_args_list[0][0][0]) == [1, 2, 3]
    assert mock_plt.bar.call_args_list[0][0][1] == [17, 19, 13]
    assert mock_plt.bar.call_args_list[0][1] == {'width': 0.35}

    # 3-2. plt_type = 'xy', short traj
    mock_plt.reset_mock()

    results = clustering.analyze_transitions(clusters, plot_type='xy')

    mock_plt.figure.called_once()
    mock_fig.add_subplot.assert_called_once_with(111)
    mock_plt.plot.called_once()
    mock_plt.xlabel.assert_called_once_with('Time frame (ps)')
    mock_plt.ylabel.assert_called_once_with('Cluster index')
    mock_ax.yaxis.set_major_locator.assert_called_once_with(mock_obj)
    mock_plt.grid.assert_called_once()
    mock_plt.savefig.assert_called_once_with('cluster_traj.png', dpi=600)

    assert (mock_plt.plot.call_args_list[0][0][0] == np.arange(0, 98, 2)).all()
    assert (mock_plt.plot.call_args_list[0][0][1] == traj).all()

    # 3-3. plt_type = 'xy', long traj
    mock_plt.reset_mock()
    clusters = {}
    t = np.arange(2000)
    clusters[1] = np.random.choice(t, 1000, replace=False)
    clusters[1] = np.sort(clusters[1])
    clusters[2] = np.array([t[i] for i in range(2000) if i not in clusters[1]])

    results = clustering.analyze_transitions(clusters, plot_type='xy')
    mock_plt.xlabel.assert_called_once_with('Time frame (ns)')
    assert (mock_plt.plot.call_args_list[0][0][0] == np.arange(2000) / 1000).all()
    
    # 3-4. invalid plt_type
    with pytest.raises(ValueError, match='Invalid plot type: test. The plot type must be either "bar" or "xy" or unspecified.'):
        clustering.analyze_transitions(clusters, plot_type='test')
