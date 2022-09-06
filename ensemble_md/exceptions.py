####################################################################
#                                                                  #
#    ensemble_md,                                                  #
#    a python package for running GROMACS simulation ensembles     #
#                                                                  #
#    Written by Wei-Tse Hsu <wehs7661@colorado.edu>                #
#    Copyright (c) 2022 University of Colorado Boulder             #
#                                                                  #
####################################################################

class ParameterError(Exception):
    """Error raised when detecting improperly specified parameters in the YAML file."""


class ParseError(Exception):
    """Error raised when parsing of a file failed. Modified from GromacsWrapper."""
