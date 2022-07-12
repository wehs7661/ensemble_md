####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package to visualize the results obtained from MD    #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2021 University of Colorado Boulder             #
#                                                                  #
####################################################################
"""
Unit tests for the module `ensemble_md.utils`.
"""
import sys
import ensemble_md.utils as utils


def test_Logger():
    sys.stdout = utils.Logger(logfile='test.txt')
    print('Test')

    f = open('test.txt', 'r')
    lines = f.readlines()
    f.close()

    assert lines[0] == 'Test\n'
