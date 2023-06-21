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
Unit tests for the module utils.py.
"""
import sys
import tempfile
import numpy as np
from ensemble_md.utils import utils


def test_logger():
    # Create a temporary file for the log
    with tempfile.TemporaryFile(mode="w+t") as log_file:
        # Get the file path for the temporary file
        log_path = log_file.name

        # Create a logger that redirects output to the temporary file
        logger = utils.Logger(log_path)

        # Redirect stdout to the logger
        sys.stdout = logger

        # Write some messages to stdout
        print("Hello, world!")
        print("Testing logger...")

        # Flush the logger to ensure that all messages are written to the log
        logger.flush()

        # Reset stdout to the original stream
        sys.stdout = sys.__stdout__


def test_format_time():
    assert utils.format_time(0) == "0.0 second(s)"
    assert utils.format_time(1) == "1.0 second(s)"
    assert utils.format_time(59) == "59.0 second(s)"
    assert utils.format_time(60) == "1 minute(s) 0 second(s)"
    assert utils.format_time(61) == "1 minute(s) 1 second(s)"
    assert utils.format_time(3599) == "59 minute(s) 59 second(s)"
    assert utils.format_time(3600) == "1 hour(s) 0 minute(s) 0 second(s)"
    assert utils.format_time(3661) == "1 hour(s) 1 minute(s) 1 second(s)"
    assert utils.format_time(86399) == "23 hour(s) 59 minute(s) 59 second(s)"
    assert utils.format_time(86400) == "1 day, 0 hour(s) 0 minute(s) 0 second(s)"
    assert utils.format_time(90061) == "1 day, 1 hour(s) 1 minute(s) 1 second(s)"


def test_autoconvert():
    # Test non-string input
    assert utils.autoconvert(42) == 42

    # Test string input that can be converted to int
    assert utils.autoconvert("42") == 42

    # Test string input that can be converted to float
    assert utils.autoconvert("3.14159") == 3.14159

    # Test string input that can be converted to a numpy array of ints
    assert utils.autoconvert("1 2 3") == [1, 2, 3]

    # Test string input that can be converted to a numpy array of floats
    assert utils.autoconvert("1.0 2.0 3.0") == [1.0, 2.0, 3.0]


def test_get_subplot_dimension():
    assert utils.get_subplot_dimension(1) == (1, 1)
    assert utils.get_subplot_dimension(2) == (1, 2)
    assert utils.get_subplot_dimension(3) == (2, 2)
    assert utils.get_subplot_dimension(4) == (2, 2)
    assert utils.get_subplot_dimension(5) == (2, 3)
    assert utils.get_subplot_dimension(6) == (2, 3)
    assert utils.get_subplot_dimension(7) == (3, 3)
    assert utils.get_subplot_dimension(8) == (3, 3)
    assert utils.get_subplot_dimension(9) == (3, 3)


def test_weighted_mean():
    # 1. Same error for each sample --> weighted mean reduces to simple mean
    vals = [1, 2, 3, 4]
    errs = [0.1, 0.1, 0.1, 0.1]
    mean, err = utils.weighted_mean(vals, errs)
    assert mean == 2.5
    assert err == np.sqrt(4 * 0.1 ** 2) / 4  # 0.05

    # 2. Different errors
    vals = [1, 2, 3, 4]
    errs = [5, 0.1, 0.1, 0.1]
    mean, err = utils.weighted_mean(vals, errs)
    assert np.isclose(mean, 2.9997333688841485)
    assert np.isclose(err, 0.0577311783020254)
