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
Unit tests for the module gmx_parser.py. Note that most part of the functions
here have been automatically tested when testing some of the functions in ensemble_md,
so here we are just testing the untested part.
"""
import os
import pytest
from ensemble_md.utils import gmx_parser
from ensemble_md.utils.exceptions import ParseError

current_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_path, "data")


def test_parse_log():
    """
    The untested cases include:
      - Case 2: The weights were equilibrated during the simulation.
      - Case 3: The weights were fixed in the simulation.
    """
    # Now let's test Case 2
    # Case 2-1: equil_time < 1000 ps
    # As a note, here we used a very lenient weight-equil-wl-delta as 0.3, with nst_sim = 2000.
    # The log file was from sim_0/iteration_1.
    wl_delta_1, weights_1, counts_1, equil_time_1 = gmx_parser.parse_log(os.path.join(input_path, 'case2_1.log'))
    assert wl_delta_1 == 0.32
    assert weights_1 == [0.00000, 1.40453, 2.85258, 2.72480, 3.46220, 5.88607]
    assert counts_1 == [11, 13, 15, 4, 4, 50]
    assert equil_time_1 == 6.0600000000000005

    # Case 2-2: equil_time > 1000 ps
    # As a note, here we used weight-equil-wl-delta as 0.3, with nst_sim = 2000,  n_iterations = 1000, n_sim = 2,
    # s = 4, wl-scale = 0.9, and wl-ratio = 0.9. The log file was from sim_0/iteration_475 (t=1.9 ns).
    wl_delta_2, weights_2, counts_2, equil_time_2 = gmx_parser.parse_log(os.path.join(input_path, 'case2_2.log'))
    assert wl_delta_2 == 0.00099828
    assert weights_2 == [0.00000, 1.60419, 2.47705, 3.14963, 3.47840]
    assert counts_2 == [2, 3, 1, 2, 1]
    assert equil_time_2 == 1903.82

    # Here we test Case 3
    wl_delta_3, weights_3, counts_3, equil_time_3 = gmx_parser.parse_log(os.path.join(input_path, 'case3.log'))
    assert wl_delta_3 is None
    assert weights_3 == [0.00000, 1.55165, 2.55043, 3.15034, 3.26889, 4.37831, 5.28574, 3.29638, 2.22527]
    assert counts_3 == [12, 9, 10, 9, 4, 5, 1, 0, 0]
    assert equil_time_3 == 0


def test_filename():
    MDP = gmx_parser.MDP()
    with pytest.raises(ValueError, match="A file name is required because no default file name was defined."):
        MDP.filename()

    MDP._filename = 'test'
    assert MDP.filename() == 'test'


class Test_MDP:
    def test__eq__(self):
        mdp_1 = gmx_parser.MDP("ensemble_md/tests/data/expanded.mdp")
        mdp_2 = gmx_parser.MDP("ensemble_md/tests/data/expanded.mdp")
        assert mdp_1 == mdp_2

    def test_read(self):
        mdp = gmx_parser.MDP()
        f = open("fake.mdp", "a")
        f.write("TEST")
        f.close()
        with pytest.raises(ParseError, match="'fake.mdp': unknown line in mdp file, 'TEST'"):
            mdp.read('fake.mdp')
        os.remove('fake.mdp')

    def test_write(self):
        mdp = gmx_parser.MDP("ensemble_md/tests/data/expanded.mdp")
        mdp.write('test_1.mdp', skipempty=False)
        mdp.write('test_2.mdp', skipempty=True)
        os.remove('test_1.mdp')
        os.remove('test_2.mdp')
