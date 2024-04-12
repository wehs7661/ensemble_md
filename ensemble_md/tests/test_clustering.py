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


def test_count_transitions():
    # Test 1: No transitions (i.e., must be the case where there was only one cluster)
    clusters = {
        1: [0, 178, 184, 186, 300, 302, 304, 306, 308, 310, 312, 318, 362, 366, 370, 372, 374, 376, 378, 380, 382, 390, 460, 464, 468, 470, 476],  # noqa: E501
    }
    n_transitions, t_transitions = clustering.count_transitions(clusters)
    assert n_transitions == 0
    assert t_transitions == []

    # Test 2: More transitions
    clusters = {
        1: [0, 2, 4, 6, 20, 22, 24, 34, 36, 38, 62, 64, 66, 80, 82, 84, 86],
        2: [8, 10, 12, 26, 28, 30, 32, 48, 50, 52, 54, 74, 76, 78, 88, 90, 92, 94, 96],
        3: [14, 16, 18, 40, 42, 44, 46, 56, 58, 60, 68, 70, 72],
    }
    n_transitions, t_transitions = clustering.count_transitions(clusters)

    assert n_transitions == 9
    assert t_transitions == [8, 20, 26, 34, 48, 62, 74, 80, 88]
